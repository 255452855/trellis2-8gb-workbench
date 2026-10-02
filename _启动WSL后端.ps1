param(
    [string]$Distro = "Ubuntu-24.04",
    [int]$Port = 8080,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = Join-Path $root "engine\code"
$outputDir = Join-Path $codeDir "output"
$logPath = Join-Path $outputDir "wsl-backend.log"
$errorLogPath = Join-Path $outputDir "wsl-backend-error.log"
$bootLogPath = Join-Path $outputDir "启动日志.txt"
$driveLetter = $root.Substring(0, 1).ToLowerInvariant()
$linuxCodeDir = "/mnt/$driveLetter" + $codeDir.Substring(2).Replace("\", "/")
$linuxPython = "/home/ccb/trellis2-wsl-venv/bin/python"

# ── 启动日志 ────────────────────────────────────────────────
# 启动流程每一步都写进 output\启动日志.txt。
# 以前脚本一出问题就直接退出、什么都不打印，看起来就像「双击了没反应」，
# 所以这里必须留下痕迹，出问题把该文件发出来即可定位。
function Write-Boot {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    try { Add-Content -LiteralPath $bootLogPath -Value $line -Encoding UTF8 } catch { }
}

# ── 打开浏览器 ──────────────────────────────────────────────
# 1) 先用 HTTP 真取一次页面（端口通 != 页面能打开，只测端口会误判）
# 2) 三种方式依次尝试，任一成功即返回
# 3) 全失败也要把网址打印出来，绝不静默
function Open-Ui {
    param([int]$UiPort)
    $uiUrl = "http://127.0.0.1:$UiPort"

    try {
        $resp = Invoke-WebRequest -Uri $uiUrl -UseBasicParsing -TimeoutSec 20
        Write-Boot ("HTTP 自检 OK: {0}, {1} 字节" -f $resp.StatusCode, $resp.RawContentLength)
    } catch {
        Write-Boot ("HTTP 自检失败: {0}" -f $_.Exception.Message)
        Write-Output ("页面自检没通过（{0}），仍尝试打开浏览器。" -f $_.Exception.Message)
    }

    # 优先用默认浏览器的 exe 开一个「新窗口」。
    # 注意：浏览器已经在后台运行时，Start-Process 网址只是给现有窗口加一个标签页，
    # 不会把窗口带到前台 —— 表现就是「双击了但浏览器根本没弹出来」。
    try {
        $cmd = (Get-Item "Registry::HKEY_CLASSES_ROOT\http\shell\open\command").GetValue("")
        if ($cmd -match '"([^"]+\.exe)"') {
            $exe = $Matches[1]
            $browserArgs = @()
            $leaf = [System.IO.Path]::GetFileName($exe).ToLowerInvariant()
            if ($leaf -eq "msedge.exe" -or $leaf -eq "chrome.exe") {
                $browserArgs += "--new-window"
            }
            $browserArgs += $uiUrl
            Start-Process -FilePath $exe -ArgumentList $browserArgs
            Write-Boot ("浏览器已打开（新窗口: {0}）" -f $exe)
            Write-Output "已在浏览器打开： $uiUrl"
            return
        }
        Write-Boot ("没解析出默认浏览器 exe，命令是: {0}" -f $cmd)
    } catch {
        Write-Boot ("新窗口方式失败: {0}" -f $_.Exception.Message)
    }

    try {
        Start-Process $uiUrl
        Write-Boot "浏览器已打开（Start-Process）"
        Write-Output "已在浏览器打开： $uiUrl"
        return
    } catch {
        Write-Boot ("Start-Process 失败: {0}" -f $_.Exception.Message)
    }

    try {
        Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "start", "", $uiUrl) -WindowStyle Hidden
        Write-Boot "浏览器已打开（cmd start）"
        Write-Output "已在浏览器打开： $uiUrl"
        return
    } catch {
        Write-Boot ("cmd start 失败: {0}" -f $_.Exception.Message)
    }

    try {
        Start-Process -FilePath "explorer.exe" -ArgumentList $uiUrl
        Write-Boot "浏览器已打开（explorer）"
        Write-Output "已在浏览器打开： $uiUrl"
        return
    } catch {
        Write-Boot ("explorer 失败: {0}" -f $_.Exception.Message)
    }

    Write-Boot "三种方式都没能打开浏览器"
    Write-Output "自动打开浏览器失败，请手动访问： $uiUrl"
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
Write-Boot "==== 启动流程开始 ===="
Write-Boot ("PowerShell {0}" -f $PSVersionTable.PSVersion)
Write-Boot ("代码目录(Windows)={0}" -f $codeDir)
Write-Boot ("代码目录(WSL)={0}" -f $linuxCodeDir)

$null = wsl.exe -d $Distro -- test -x $linuxPython
Write-Boot ("wsl test -x 退出码={0}" -f $LASTEXITCODE)
if ($LASTEXITCODE -ne 0) {
    Write-Boot "WSL Python 环境不存在"
    throw "WSL Python 环境不存在：$linuxPython"
}

$null = wsl.exe -d $Distro -- test -f "$linuxCodeDir/app.py"
Write-Boot ("wsl test -f 退出码={0}" -f $LASTEXITCODE)
if ($LASTEXITCODE -ne 0) {
    Write-Boot "WSL 无法访问应用文件"
    throw "WSL 无法访问应用文件：$linuxCodeDir/app.py"
}

# ── -Restart：先干掉旧后端，再起新的 ────────────────────────
# 改了 app.py 之后必须让新代码生效，而「端口已被占用」分支只会打开旧页面，
# 所以这里提供一个显式的重启入口。
if ($Restart) {
    Write-Boot "收到 -Restart：先停掉旧后端"
    try {
        # pkill 用 "ap[p].py" 这种写法，否则 pkill 会匹配到自己所在的命令行而自杀
        $null = wsl.exe -d $Distro -- pkill -f "ap[p].py"
        Write-Boot ("pkill app.py 退出码={0}" -f $LASTEXITCODE)
        # 再补一刀：worker 是 multiprocessing 子进程，父进程被强杀时可能变成孤儿，
        # 孤儿会一直占着显存（8GB 卡上这个很致命）
        Start-Sleep -Seconds 2
        $null = wsl.exe -d $Distro -- pkill -f "trellis2-wsl-venv/bin/python"
        Write-Boot ("pkill venv python 退出码={0}" -f $LASTEXITCODE)
    } catch {
        Write-Boot ("pkill 失败: {0}" -f $_.Exception.Message)
    }
    # 必须等端口真正释放：否则新后端会自己找下一个端口，而本脚本还在等 8080
    $freeDeadline = (Get-Date).AddSeconds(40)
    while ((Get-Date) -lt $freeDeadline) {
        Start-Sleep -Seconds 1
        $still = $false
        try {
            $still = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -InformationLevel Quiet
        } catch { }
        if (-not $still) { break }
    }
    Write-Boot "旧后端已停止，端口应已释放"
}

# 端口已经在监听：说明后端本来就在跑，直接开页面，别再起第二个
$already = $false
try {
    $already = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -InformationLevel Quiet
} catch {
    Write-Boot ("端口检测失败: {0}" -f $_.Exception.Message)
}
Write-Boot ("端口 {0} 已在监听={1}" -f $Port, $already)

if ($already) {
    Write-Boot "分支：已有服务在跑，直接打开页面"
    Write-Output "已有服务正在 127.0.0.1:$Port 运行，正在打开页面…"
    Write-Output "（改了代码想让新代码生效，请改用「重启后端.bat」）"
    Open-Ui -UiPort $Port
    exit 0
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$arguments = @(
    "-d", $Distro,
    "--cd", $linuxCodeDir,
    "--exec", "env",
    "CUDA_HOME=/usr/local/cuda-12.8",
    "PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib",
    "LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64",
    "TORCH_CUDA_ARCH_LIST=6.1",
    "XFORMERS_IGNORE_FLASH_VERSION_CHECK=1",
    "ATTN_BACKEND=sdpa",
    # ⚠️ 必须在**启动时**就设好：这个变量在 CUDA 分配器初始化时被读取一次，
    # 之后再改 os.environ 可能已经不生效了。以前 app.py 设 max_split_size_mb:128，
    # 那是给旧分配器的补丁，会阻止大块复用、把显存顶到上限（2026-10-02 实测
    # HR 阶段顶到 7842/8192MiB → WSL 驱动超额订阅 → dmesg make_resident ENOMEM → 颠簸）。
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.65",
    "OPENCV_IO_ENABLE_OPENEXR=1",
    "HF_HOME=$linuxCodeDir/MODELS",
    "HF_HUB_OFFLINE=1",
    "PYTHONUNBUFFERED=1",
    $linuxPython,
    "app.py",
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--no-browser"
)

# 只给「含空格」的参数加引号。
# ⚠️ 千万别给每个参数都套引号 —— wsl.exe 看到命令行以 " 开头就不认 -d / --cd 这些选项了，
# 会把 "-d" 当成要执行的命令丢给默认 shell，报
#   /bin/bash: line 1: -d: command not found
# 然后秒退（wsl-backend.log 是空的，wsl-backend-error.log 里就那一行）。
$quotedArguments = $arguments | ForEach-Object {
    $a = [string]$_
    if ($a -match '\s') { '"' + $a.Replace('"', '\"') + '"' } else { $a }
}

Write-Boot "启动 wsl.exe ..."
try {
    $process = Start-Process `
        -FilePath "wsl.exe" `
        -ArgumentList ($quotedArguments -join " ") `
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

$deadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    $up = $false
    try {
        $up = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port -InformationLevel Quiet
    } catch {
        Write-Boot ("端口轮询失败: {0}" -f $_.Exception.Message)
    }
    if ($up) {
        Write-Boot "端口已就绪，准备打开浏览器"
        Write-Output "WSL 后端已启动： http://127.0.0.1:$Port"
        Write-Output "日志：$logPath"
        Open-Ui -UiPort $Port
        exit 0
    }
    if ($process.HasExited) {
        # 失败详情必须写进启动日志，否则用户只看到窗口一闪而过，什么线索都没有
        Write-Boot "wsl.exe 提前退出，后端没起来 —— 下面是日志尾部"
        $details = @()
        foreach ($path in @($logPath, $errorLogPath)) {
            Write-Boot ("--- {0} ---" -f $path)
            if (Test-Path -LiteralPath $path) {
                $tail = @(Get-Content -LiteralPath $path -Tail 40)
                if ($tail.Count -eq 0) {
                    Write-Boot "    (空文件)"
                }
                foreach ($line in $tail) {
                    $details += $line
                    Write-Boot ("    " + $line)
                }
            } else {
                Write-Boot "    (文件不存在)"
            }
        }
        if ($details.Count -eq 0) {
            $details = @("没有生成启动日志。")
        }
        throw "WSL 后端启动失败。`n$($details -join [Environment]::NewLine)"
    }
}

Write-Boot "等待端口超时（5 分钟）"
throw "等待 WSL 后端超过 5 分钟。日志：$logPath"
