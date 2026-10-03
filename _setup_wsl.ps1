# WSL2 引导安装 —— 被 _start.ps1 调用，也可以单独双击运行查看检查结果。
#
# 为什么需要它：以前假定「机器上已经有 WSL + Ubuntu + nvidia 驱动」，
# 别人拿到包后 wsl.exe 都不存在，直接抛一句英文错误就退出了。
# 现在这里负责把「WSL 没装 / 发行版没装 / 需要重启」三种情况区分开，
# 每种都给可执行的下一步。
#
# 退出码约定（_start.ps1 按这个决定流程）：
#   0 = WSL 和目标发行版都可用
#   9 = 已启用组件，但必须重启 Windows 后再接着跑
#   1 = 装不上，需要人工介入（输出里有原因）
#
# 注意：wsl.exe 无论控制台代码页是什么都输出 UTF-16LE，PowerShell 默认按
# ANSI 解码会得到「一 个 字 一 个 空 格」的乱码，所以调用前后要临时切
# [Console]::OutputEncoding。
param(
    [string]$Distro = 'Ubuntu-24.04',
    [string]$BootLog = '',
    [switch]$Quiet
)

$ErrorActionPreference = 'Continue'
$SelectedDistro = $Distro      # 可能被改成本机已有的另一个 Ubuntu
$prevEncoding = $null
try { $prevEncoding = [Console]::OutputEncoding } catch { }

function Write-Boot {
    param([string]$Message)
    if ($BootLog -eq '') { return }
    $line = "[{0}] [wsl] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    try { Add-Content -LiteralPath $BootLog -Value $line -Encoding UTF8 } catch { }
}

function Say {
    param([string]$Message)
    Write-Boot $Message
    if (-not $Quiet) { Write-Host $Message }
}

# 调 wsl.exe 并拿回退出码 + 文本。
# 编码/分流的原因见 _start.ps1 里的 Invoke-WslText（同一套实测结论）：
# Linux 侧输出是 UTF-8，wsl.exe 自己的警告是另一套编码，混在一起会把
# `wsl -l -q` 的发行版名单污染成"一条假发行版"。
function Invoke-Wsl {
    param([string[]]$WslArgs)
    $result = @{ Code = 1; Text = ''; Err = '' }
    $eap = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        $raw = & wsl.exe @WslArgs 2>&1
        $result.Code = $LASTEXITCODE
        $result.Text = (@($raw | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $null } else { [string]$_ } }) -join "`n").Trim()
        $result.Err = (@($raw | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $null } }) -join "`n").Trim()
    } catch {
        $result.Code = 1
        $result.Err = $_.Exception.Message
    } finally {
        $ErrorActionPreference = $eap
        if ($null -ne $prevEncoding) {
            try { [Console]::OutputEncoding = $prevEncoding } catch { }
        }
    }
    return $result
}

Say "---- WSL 环境检查（目标发行版：$Distro）----"

# ── 1. 操作系统版本 ────────────────────────────────────────────────
# WSL2 需要 Win10 2004（19041）及以上；更早的版本只有 WSL1，跑不了 CUDA。
$build = [System.Environment]::OSVersion.Version.Build
$release = (Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' `
             -ErrorAction SilentlyContinue).DisplayVersion
if ($build -lt 19041) {
    Say "[失败] Windows build $build 太旧，WSL2 需要 Windows 10 2004(19041) 或更高（当前 $release）"
    Write-Host "  请先把 Windows 更新到 21H2 以上，再重跑 start.bat。" -ForegroundColor Red
    exit 1
}
Say "系统版本 OK：Windows build $build (版本 $release)"

# ── 2. 虚拟化是否可用（只警告，不阻断：有些机器查不到该字段）────────
try {
    $cpu = Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1
    if ($null -ne $cpu.VirtualizationFirmwareEnabled -and -not $cpu.VirtualizationFirmwareEnabled) {
        Say "[警告] BIOS 里的虚拟化（VT-x / AMD-V）看起来是关闭的，WSL2 可能起不来"
        Write-Host "  若下一步报 Wsl/Service/RegisterDistro/CreateInstance 0xE8000006 之类错误，" -ForegroundColor Yellow
        Write-Host "  请进 BIOS 打开 Virtualization / VT-x 后重试。" -ForegroundColor Yellow
    }
} catch {
    Say "[警告] 读不到 CPU 虚拟化状态，跳过该检查"
}

# ── 3. wsl.exe 在不在 ──────────────────────────────────────────────
$wslExe = Join-Path $env:SystemRoot "System32\wsl.exe"
if (-not (Test-Path -LiteralPath $wslExe)) {
    Say "[失败] 系统里没有 wsl.exe，WSL 组件从未安装"
    Write-Host "  请以管理员身份打开 PowerShell，执行：" -ForegroundColor Yellow
    Write-Host "      wsl --install" -ForegroundColor Cyan
    Write-Host "  然后重启，再双击 start.bat。" -ForegroundColor Yellow
    exit 1
}

# ── 4. 目标发行版是否已经能用 ──────────────────────────────────────
# 不用解析 `wsl -l -v` 的文本（本地化 + UTF-16 + 列宽都可能变），
# 直接问 wsl：能不能在这个发行版里跑一条命令。
$probe = Invoke-Wsl @('-d', $Distro, '--', 'true')
if ($probe.Code -eq 0) {
    Say "发行版可用：$Distro"
} else {
    Say "发行版不可用（退出码 $($probe.Code)）：$($probe.Text)"
    # 看看是不是只是名字写错了：列出已装的发行版
    $list = Invoke-Wsl @('-l', '-q')
    $installed = @()
    if ($list.Code -eq 0) {
        $installed = @($list.Text -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })
    }
    if ($installed.Count -gt 0) {
        Say "本机已安装的发行版：$($installed -join ', ')"
        # 有别的 Ubuntu 就复用它，别逼用户再装一遍 20GB
        $alt = $installed | Where-Object { $_ -match 'ubuntu' } | Select-Object -First 1
        if ($alt) {
            Say "改用已安装的 $alt（写进 runtime.conf，之后不用再选）"
            $r = Invoke-Wsl @('-d', $alt, '--', 'true')
            if ($r.Code -eq 0) {
                $SelectedDistro = $alt
                Say "发行版可用：$alt"
            }
        }
    }
    if ($SelectedDistro -eq $Distro) {
        # 需要装发行版：先确认 WSL 引擎本身可用
        $engine = Invoke-Wsl @('--status')
        Say "--status 退出码 $($engine.Code)：$($engine.Text)"
        if ($engine.Code -ne 0) {
            Say "WSL 组件尚未启用，尝试安装（会弹 UAC 提权）"
            $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
            if (-not $admin) {
                Say "当前不是管理员，wsl --install 可能需要你点一下 UAC 提权框"
            }
            $r = Invoke-Wsl @('--install', '--no-launch')
            Say "wsl --install --no-launch 退出码 $($r.Code)：$($r.Text)"
            Write-Host ""
            Write-Host "  WSL 组件已请求安装。请重启 Windows，然后再次双击 start.bat。" -ForegroundColor Cyan
            Write-Host "  重启后第一次进入 Ubuntu 时，会让你设置 Linux 用户名和密码（随便设，记牢）。" -ForegroundColor Cyan
            Say "已请求安装 WSL 组件，等待用户重启"
            exit 9
        }
        # 引擎在，只差发行版
        Say "WSL 引擎正常，安装发行版 $Distro（约 1~3GB，需要联网）"
        $r = Invoke-Wsl @('--install', '-d', $Distro, '--no-launch')
        if ($r.Code -ne 0) {
            Say "--no-launch 不被支持或失败（$($r.Code)）：$($r.Text)，改跑 wsl --install -d $Distro"
            $r2 = Invoke-Wsl @('--install', '-d', $Distro)
            Say "wsl --install -d 退出码 $($r2.Code)：$($r2.Text)"
            if ($r2.Code -ne 0) {
                Write-Host "  [失败] 装不上 $Distro。请手动执行： wsl --install -d Ubuntu-24.04" -ForegroundColor Red
                exit 1
            }
        }
        $probe = Invoke-Wsl @('-d', $Distro, '--', 'true')
        if ($probe.Code -ne 0) {
            Write-Host ""
            Write-Host "  发行版已下载，但还没有完成首次初始化。" -ForegroundColor Cyan
            Write-Host "  请在 PowerShell 里执行一次： wsl -d $Distro" -ForegroundColor Cyan
            Write-Host "  按提示设置 Linux 用户名/密码，然后再双击 start.bat。" -ForegroundColor Cyan
            Say "发行版需要人工完成首次初始化（设置用户名）"
            exit 9
        }
        Say "发行版可用：$Distro"
    }
}

# ── 5. 默认版本必须是 2（WSL1 没有 /dev/dxg，用不了 CUDA）──────────
$null = Invoke-Wsl @('--set-default-version', '2')

# 把最终选定的发行版回传给调用方（Write-Host 不进管道，这里只有一行输出）
"$SelectedDistro"
Say "WSL 检查通过"
exit 0
