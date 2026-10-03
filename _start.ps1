# ─────────────────────────────────────────────────────────────────────
# 一键包的 Windows 侧大脑：检查 → （缺就）安装 → 起后端 → 开浏览器
#
# 由 start.bat 调用，也可以单独跑：
#   powershell -ExecutionPolicy Bypass -File _start.ps1
#   powershell -ExecutionPolicy Bypass -File _start.ps1 -Restart     # 改了代码后重启后端
#   powershell -ExecutionPolicy Bypass -File _start.ps1 -Status      # 只报告状态，不改动
#   powershell -ExecutionPolicy Bypass -File _start.ps1 -Install     # 强制补装
#   powershell -ExecutionPolicy Bypass -File _start.ps1 -NoFlux      # 跳过文生3D（省 23GB）
#
# 它自己不装任何东西 —— 安装全部交给 WSL 侧的 _oneclick.sh，因为 nvcc、
# pip、CUDA 扩展编译都发生在 Linux 里。本脚本只负责：WSL 本身是否存在、
# 显卡在 WSL 里可见吗、端口起来了吗、浏览器打开。
#
# 全流程每一步都写进 engine\code\output\boot-log.txt —— 双击后「窗口一闪
# 而过」时这是唯一的线索，出问题把该文件发出来就能定位。（文件名是 ASCII：
# .bat 里打不出中文路径，报错提示必须能原样引用它。）
#
# ⚠️ 本文件带 UTF-8 BOM：PowerShell 5.1 读无 BOM 的 UTF-8 会按 ANSI 解码，
#    中文注释变成非法字符后连字符串都不一定闭合，脚本会以看不懂的错误退出。
#    （PowerShell 7 默认按 UTF-8 读，不带 BOM 也没事；这里是向后兼容 Win10 自带的 5.1。）
# ─────────────────────────────────────────────────────────────────────
param(
    [switch]$Restart,
    [switch]$Install,
    [switch]$Status,
    [switch]$NoFlux,
    [int]$Port = 0,
    [string]$Distro = ''
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$codeDir = Join-Path $root 'engine\code'
$outputDir = Join-Path $codeDir 'output'
$logPath = Join-Path $outputDir 'wsl-backend.log'
$errorLogPath = Join-Path $outputDir 'wsl-backend-error.log'
$bootLogPath = Join-Path $outputDir 'boot-log.txt'
$confPath = Join-Path $root 'runtime.conf'
$script:WslCode = 1

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

# ── 工具函数（必须定义在调用之前，PowerShell 没有函数提升）──────────
function Write-Boot {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    try { Add-Content -LiteralPath $bootLogPath -Value $line -Encoding UTF8 } catch { }
}

# runtime.conf 是 WSL 侧安装时写的 key=value，Windows 侧也要读。
# 只做纯文本解析，绝不 Invoke-Expression（值里可能有空格/反引号）。
function Read-Conf {
    $map = @{}
    if (Test-Path -LiteralPath $confPath) {
        foreach ($raw in Get-Content -LiteralPath $confPath -Encoding UTF8) {
            $line = $raw.Trim()
            if ($line -eq '' -or $line.StartsWith('#')) { continue }
            $i = $line.IndexOf('=')
            if ($i -lt 1) { continue }
            $map[$line.Substring(0, $i).Trim().ToLowerInvariant()] = $line.Substring($i + 1).Trim()
        }
    }
    return $map
}

# 端口通 != 页面能打开，但作为"后端起来了"的第一信号够用；
# -WarningAction 是为了别把 Test-NetConnection 的警告刷到用户屏幕上。
function Test-PortOpen {
    param([int]$ProbePort)
    try {
        return (Test-NetConnection -ComputerName 127.0.0.1 -Port $ProbePort `
                -InformationLevel Quiet -WarningAction SilentlyContinue)
    } catch { return $false }
}

# 调 wsl.exe 并拿回 stdout 文本；退出码写进 $script:WslCode，stderr 写进 $script:WslErr。
#
# 两个坑都是实测踩出来的（.testrun/15-capture.log、16-paths.log）：
#  1) Linux 侧的输出是 **UTF-8**，所以临时把 [Console]::OutputEncoding 设成 UTF8；
#     设成 Unicode(UTF-16) 会得到 "洯瑮搯"（每个真实字符后面其实没有 0x00）。
#  2) wsl.exe 自己的引擎警告（本机 .wslconfig 配了镜像网络，每次都吐一行
#     "wsl: 使用镜像网络模式时…"）是 Windows 侧写的 UTF-16，混进 stdout 之后
#     "结果必须以 / 开头"这类判断全错。所以按对象类型把 stdout 和 stderr 分开，
#     stderr 单独留作诊断，不参与解析。
function Invoke-WslText {
    param([string[]]$WslArgs)
    $script:WslCode = 1
    $script:WslErr = ''
    $prev = $null
    $eap = $ErrorActionPreference
    $outLines = @()
    try {
        $ErrorActionPreference = 'Continue'   # 原生化 stderr 成 ErrorRecord，别让它中断脚本
        $prev = [Console]::OutputEncoding
        [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
        $raw = & wsl.exe @WslArgs 2>&1
        $script:WslCode = $LASTEXITCODE
        $outLines = @($raw | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $null } else { [string]$_ } })
        $script:WslErr = (@($raw | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { $null } }) -join "`n")
    } catch {
        $script:WslErr = $_.Exception.Message
    } finally {
        $ErrorActionPreference = $eap
        if ($null -ne $prev) { try { [Console]::OutputEncoding = $prev } catch { } }
    }
    return ($outLines -join "`n").Trim()
}

# 打开浏览器：先 HTTP 真取一次页面，再用四种方式依次尝试。
# 全失败也必须把网址打印出来，绝不静默 —— 否则用户以为程序没跑起来。
function Open-Ui {
    param([int]$UiPort)
    $uiUrl = "http://127.0.0.1:$UiPort"

    try {
        $resp = Invoke-WebRequest -Uri $uiUrl -UseBasicParsing -TimeoutSec 20
        Write-Boot ("HTTP 自检 OK: {0}, {1} 字节" -f $resp.StatusCode, $resp.RawContentLength)
    } catch {
        Write-Boot ("HTTP 自检失败: {0}" -f $_.Exception.Message)
        Write-Host ("  页面自检没通过（{0}），仍尝试打开浏览器。" -f $_.Exception.Message) -ForegroundColor Yellow
    }

    # 优先用默认浏览器的 exe 开一个「新窗口」。浏览器已在后台运行时，
    # Start-Process 网址只是加个标签页、不会带到前台 ——
    # 表现就是「双击了但浏览器根本没弹出来」。
    try {
        $cmd = (Get-Item "Registry::HKEY_CLASSES_ROOT\http\shell\open\command").GetValue("")
        if ($cmd -match '"([^"]+\.exe)"') {
            $exe = $Matches[1]
            $browserArgs = @()
            $leaf = [System.IO.Path]::GetFileName($exe).ToLowerInvariant()
            if ($leaf -eq 'msedge.exe' -or $leaf -eq 'chrome.exe') { $browserArgs += '--new-window' }
            $browserArgs += $uiUrl
            Start-Process -FilePath $exe -ArgumentList $browserArgs
            Write-Boot ("浏览器已打开（新窗口: {0}）" -f $exe)
            Write-Host ("  已在浏览器打开： {0}" -f $uiUrl) -ForegroundColor Green
            return
        }
        Write-Boot ("没解析出默认浏览器 exe，命令是: {0}" -f $cmd)
    } catch { Write-Boot ("新窗口方式失败: {0}" -f $_.Exception.Message) }

    foreach ($way in @('Start-Process', 'cmd start', 'explorer')) {
        try {
            switch ($way) {
                'Start-Process' { Start-Process $uiUrl }
                'cmd start'     { Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', 'start', '', $uiUrl) -WindowStyle Hidden }
                'explorer'      { Start-Process -FilePath 'explorer.exe' -ArgumentList $uiUrl }
            }
            Write-Boot ("浏览器已打开（{0}）" -f $way)
            Write-Host ("  已在浏览器打开： {0}" -f $uiUrl) -ForegroundColor Green
            return
        } catch { Write-Boot ("$way 失败: {0}" -f $_.Exception.Message) }
    }

    Write-Boot "四种方式都没能打开浏览器"
    Write-Host "  自动打开浏览器失败，请手动访问： $uiUrl" -ForegroundColor Yellow
}

# ── 0. 读机器本地配置 ──────────────────────────────────────────────
$conf = Read-Conf
if (-not $Distro) { $Distro = $conf['distro'] }
if (-not $Distro) { $Distro = 'Ubuntu-24.04' }
if ($Port -eq 0) {
    if ($conf['port'] -and $conf['port'] -match '^\d+$') { $Port = [int]$conf['port'] } else { $Port = 8080 }
}

Write-Boot "==== 一键启动开始（模式：$(if ($Status) { 'status' } elseif ($Restart) { 'restart' } else { 'start' })） ===="
Write-Boot ("PowerShell {0} / Windows build {1}" -f $PSVersionTable.PSVersion, [System.Environment]::OSVersion.Version.Build)
Write-Boot ("脚本目录(Windows)=$root  候选发行版=$Distro  端口=$Port")

# ── 1. WSL 与发行版 ────────────────────────────────────────────────
Write-Host ""
Write-Host "  [1/4] 检查 WSL 环境…" -ForegroundColor Cyan
$bootstrap = Join-Path $root '_setup_wsl.ps1'
if (-not (Test-Path -LiteralPath $bootstrap)) {
    throw "缺少 _setup_wsl.ps1，项目文件不完整。"
}
$bootstrapOut = @(& $bootstrap -Distro $Distro -BootLog $bootLogPath)
$bootstrapCode = $LASTEXITCODE
# 子脚本最后一行返回它最终选定的发行版（本机可能装的是 Ubuntu 而非 Ubuntu-24.04）
$picked = $bootstrapOut | Where-Object { "$_" -match '^[A-Za-z0-9][A-Za-z0-9._\-]*$' } | Select-Object -Last 1
if ($picked) { $Distro = "$picked".Trim() }
Write-Boot ("_setup_wsl.ps1 退出码=$bootstrapCode 选定发行版=$Distro")
if ($bootstrapCode -ne 0) {
    if ($bootstrapCode -eq 9) {
        Write-Host ""
        Write-Host "  需要重启 Windows 才能继续。重启后再双击 start.bat 即可，已装好的部分不会重复装。" -ForegroundColor Yellow
        exit 9
    }
    Write-Host "  WSL 环境检查未通过，见上面的提示。" -ForegroundColor Red
    exit 1
}

# ── 2. 项目目录：换算成 WSL 里的路径 ────────────────────────────────
# 以前是 $root.Substring(0,1) 手拼，盘符一变（装到 E: 或网络盘）就全盘失效。
# 也不用 `wslpath -u $root`：实测（.testrun/16-paths.log）PowerShell 传参时
# 反斜杠会被吃掉（D:\IDM\TRELLIS2 → D:IDMTRELLIS2，rc=1），路径不带空格时才
# 出问题，带空格反而"碰巧"能过。可靠的做法是让 wsl.exe 自己 --cd 过去再 pwd：
# 反斜杠、空格、中文目录名都验证通过。
$linuxRoot = Invoke-WslText @('-d', $Distro, '--cd', $root, '--', 'pwd')
if ($script:WslCode -ne 0 -or $linuxRoot -notmatch '^/') {
    # 兜底：手拼 /mnt/<盘符>（只处理最常见的本地盘）
    if ($root -match '^([A-Za-z]):\\?(.*)$') {
        $linuxRoot = "/mnt/" + $Matches[1].ToLowerInvariant() + ($Matches[2] -replace '\\', '/')
    }
}
$linuxRoot = $linuxRoot.TrimEnd('/')
if ($linuxRoot -notmatch '^/') {
    Write-Boot ("路径换算失败：code=$($script:WslCode) out=$linuxRoot stderr=$($script:WslErr)")
    throw "WSL 无法进入项目目录 $root。如果刚从压缩包解压出来，先确认文件完整、没有被安全软件拦下。"
}
Write-Boot ("项目目录(WSL)=$linuxRoot")
Invoke-WslText @('-d', $Distro, '--cd', $root, '--', 'test', '-f', './_lib.sh') | Out-Null
if ($script:WslCode -ne 0) {
    Write-Boot "仓库文件不完整：找不到 _lib.sh"
    throw "WSL 里看不到 $linuxRoot/_lib.sh —— 项目文件不完整，或被杀毒软件隔离了。"
}

# ── 3. 显卡在 WSL 里可见吗 ──────────────────────────────────────────
$gpu = Invoke-WslText @('-d', $Distro, '--', 'nvidia-smi', '-L')
if ($script:WslCode -ne 0) {
    Write-Boot ("WSL 内 nvidia-smi 失败：$gpu")
    Write-Host ""
    Write-Host "  [失败] WSL 里看不到 NVIDIA 显卡。请：" -ForegroundColor Red
    Write-Host "    1) 把 GeForce 驱动升到较新版本（Win10 以上的通用驱动自带 WSL 支持，不需要装 Linux 驱动）"
    Write-Host "    2) 完全退出杀毒/安全软件，然后在 PowerShell 里跑一次 'wsl --shutdown'"
    Write-Host "    3) 重新双击 start.bat"
    exit 1
}
$gpuLine = (($gpu -split "`r?`n" | Where-Object { $_.Trim() -ne '' } | Select-Object -First 1)).Trim()
Write-Host "  显卡：$gpuLine" -ForegroundColor Green
Write-Boot "WSL 显卡：$($gpu -join ' | ')"

if ($Status) {
    Write-Host ""
    Write-Host "  [2/4] 环境状态（--status，不做任何改动）" -ForegroundColor Cyan
    $st = Invoke-WslText @('-d', $Distro, '--cd', $linuxRoot, '--', 'bash', './_oneclick.sh', '--status')
    Write-Host $st
    Write-Boot "--status 输出：$st"
    exit 0
}

# ── 4. 缺什么就先装什么 ────────────────────────────────────────────
Write-Host ""
Write-Host "  [2/4] 检查环境是否装齐（首次会自动安装，可能要 1~2 小时）…" -ForegroundColor Cyan
$checkOut = Invoke-WslText @('-d', $Distro, '--cd', $linuxRoot, '--', 'bash', './_oneclick.sh', '--check')
$checkCode = $script:WslCode
Write-Boot ("_oneclick.sh --check 退出码=$checkCode 输出=$checkOut")
if ($checkOut) { Write-Host "  $checkOut" }

# 退出码：0 就绪 / 1 缺环境 / 2 缺权重
if (($checkCode -ne 0) -or $Install) {
    Write-Host ""
    Write-Host "  ── 开始自动安装。进度会一直打印，中途别关窗口 ──" -ForegroundColor Yellow
    $installArgs = @('-d', $Distro, '--cd', $linuxRoot, '--', 'bash', './_oneclick.sh', '--yes')
    if ($NoFlux) { $installArgs += '--no-flux' }
    # 安装直接跑在当前控制台：里面有 apt 交互、sudo 密码、几十 GB 的下载进度，
    # 藏到后台窗口里用户只会看到「卡住了」。
    # 这里**不改** [Console]::OutputEncoding：这段输出不经管道，是让 conhost 直接
    # 渲染的；设成 UTF8 反而会让 PowerShell 自己的中文提示变成乱码。
    try {
        & wsl.exe @installArgs
        $installCode = $LASTEXITCODE
    } catch {
        Write-Boot ("安装阶段异常: {0}" -f $_.Exception.Message)
        $installCode = 1
    }
    Write-Boot ("_oneclick.sh --yes 退出码=$installCode")
    if ($installCode -ne 0) {
        Write-Host ""
        Write-Host "  安装没有全部完成（退出码 $installCode）。上面最后几行就是原因，" -ForegroundColor Red
        Write-Host "  修好后重跑本脚本即可 —— 已完成的部分会自动跳过。" -ForegroundColor Red
        Write-Host "  只想补某一步的话： wsl -d $Distro --cd $linuxRoot -- bash ./_oneclick.sh --only weights" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  安装完成。" -ForegroundColor Green
} else {
    Write-Host "  环境已就绪，跳过安装。" -ForegroundColor Green
}

# ── 5. 起后端 ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "  [3/4] 启动后端…" -ForegroundColor Cyan

if ($Restart) {
    Write-Boot "收到 -Restart：先停掉旧后端"
    Write-Host "  停止旧后端（会等显存真正释放）…"
    $stopOut = Invoke-WslText @('-d', $Distro, '--cd', $linuxRoot, '--', 'bash', './_stop_3d_backend.sh')
    Write-Boot ("_stop_3d_backend.sh：$stopOut")
}

# 端口在监听 = 后端本来就在跑，直接开页面，别再起第二个（第二个会自己换端口，
# 于是本脚本一直等 $Port，看起来像「启动卡死」）。
$already = Test-PortOpen -ProbePort $Port
Write-Boot ("端口 $Port 已在监听=$already")
if ($already) {
    if (-not $Restart) {
        Write-Host ("  已有服务在 127.0.0.1:{0} 运行，直接打开页面。" -f $Port) -ForegroundColor Green
        Write-Host "  （改了代码想让新代码生效，请用 restart-backend.bat）"
        Open-Ui -UiPort $Port
        exit 0
    }
    # 要求重启但端口没释放：等 60 秒，仍占用就说明有孤儿进程，硬起第二个后端只会更乱
    $freeDeadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $freeDeadline) {
        Start-Sleep -Seconds 2
        if (-not (Test-PortOpen -ProbePort $Port)) { break }
    }
    if (Test-PortOpen -ProbePort $Port) {
        Write-Boot "重启后端口仍被占用"
        throw "旧后端没退干净，端口 $Port 仍被占用。手动执行： wsl -d $Distro --cd $linuxRoot -- bash ./_stop_3d_backend.sh"
    }
    Write-Boot "旧后端已停止，端口已释放"
}

# 后端的环境变量全部在 WSL 侧的 _run_backend.sh 里组装（唯一真源）：
# PYTORCH_CUDA_ALLOC_CONF 必须在启动进程时就设好，CUDA 分配器只读一次，
# 之后再改 os.environ 不生效，所以绝不能在这里只传一半。
$arguments = @('-d', $Distro, '--cd', $linuxRoot, '--', 'bash', './_run_backend.sh')
# 只给「含空格」的参数加引号。千万别每个都套引号 —— wsl.exe 看到命令行以 " 开头
# 就不认 -d / --cd 了，会把 "-d" 当成要执行的命令丢给默认 shell，
# 报 "-d: command not found" 后秒退（日志是空的，最难查的那种）。
$quotedArguments = $arguments | ForEach-Object {
    $a = [string]$_
    if ($a -match '\s') { '"' + $a.Replace('"', '\"') + '"' } else { $a }
}

Write-Boot "启动 wsl.exe 后端"
try {
    $process = Start-Process -FilePath 'wsl.exe' `
        -ArgumentList ($quotedArguments -join ' ') `
        -WorkingDirectory $root `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $errorLogPath `
        -WindowStyle Hidden `
        -PassThru
} catch {
    Write-Boot ("Start-Process wsl.exe 失败: {0}" -f $_.Exception.Message)
    throw "启动 WSL 后端失败：$($_.Exception.Message)"
}
Write-Boot ("wsl.exe 已启动, PID={0}" -f $process.Id)

# 8GB 卡上模型要从 Windows 盘（drvfs）读几十 GB 权重，加载可能要几分钟，
# 给 8 分钟；超时就把日志尾部写进启动日志，别让用户对着空白窗口干等。
$deadline = (Get-Date).AddMinutes(8)
$lastTick = Get-Date
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    if (Test-PortOpen -ProbePort $Port) {
        Write-Boot "端口已就绪，准备打开浏览器"
        Write-Host ("  [4/4] 后端已启动： http://127.0.0.1:{0}" -f $Port) -ForegroundColor Green
        Open-Ui -UiPort $Port
        exit 0
    }
    if ($process.HasExited) { break }
    if (((Get-Date) - $lastTick).TotalSeconds -ge 30) {
        $lastTick = Get-Date
        Write-Host "  还在加载模型…（实时日志：$logPath）"
    }
}

Write-Boot "后端没起来 —— 下面是日志尾部"
$details = @()
foreach ($path in @($logPath, $errorLogPath)) {
    Write-Boot ("--- {0} ---" -f $path)
    if (Test-Path -LiteralPath $path) {
        $tail = @(Get-Content -LiteralPath $path -Tail 40)
        if ($tail.Count -eq 0) { Write-Boot "    (空文件)" }
        foreach ($line in $tail) { $details += $line; Write-Boot ("    " + $line) }
    } else { Write-Boot "    (文件不存在)" }
}
if ($details.Count -eq 0) { $details = @("没有生成后端日志。") }
Write-Host ""
Write-Host "  [启动失败] 后端没能起来。最近日志：" -ForegroundColor Red
foreach ($line in ($details | Select-Object -Last 15)) { Write-Host "    $line" }
Write-Host ""
Write-Host "  完整过程日志：$bootLogPath" -ForegroundColor Yellow
Write-Host "  重试或补装某一步：powershell -ExecutionPolicy Bypass -File _start.ps1 -Install" -ForegroundColor Yellow
throw "WSL 后端启动失败。`n$($details -join [Environment]::NewLine)"
