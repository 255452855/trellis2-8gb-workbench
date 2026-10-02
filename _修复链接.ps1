# venv 符号链接修复 —— venv 里的 .pyd / .dll 是指向内置 Python 的符号链接，
# 整个文件夹移动后链接会失效，导致 python.exe 起不来（报 python311.dll 找不到）。
# 本脚本把它们替换成真实文件副本，避免依赖符号链接权限。
param(
    [string]$Root = (Split-Path -Parent $MyInvocation.MyCommand.Path)
)

$ErrorActionPreference = 'Continue'
$scripts = Join-Path $Root 'engine\code\venv\Scripts'
$newRoot = Join-Path $Root 'engine\tools\python'

if (-not (Test-Path -LiteralPath $scripts)) {
    Write-Output "  跳过（找不到 $scripts）"
    exit 0
}

$fixed = 0
$broken = 0

Get-ChildItem -LiteralPath $scripts -Force -ErrorAction SilentlyContinue | ForEach-Object {
    $item = $_
    if ($item.LinkType -ne 'SymbolicLink') { return }
    $tgt = $item.Target
    if (-not $tgt) { return }

    # 只在链接已经失效（指向不存在的路径）时才修
    if (Test-Path -LiteralPath $tgt -ErrorAction SilentlyContinue) { return }

    $name = $item.Name
    $src = Join-Path $newRoot $name
    if (-not (Test-Path -LiteralPath $src)) {
        # 试着从原始目标路径提取文件名再找一次
        $src = Join-Path $newRoot (Split-Path -Leaf $tgt)
    }

    if (Test-Path -LiteralPath $src) {
        try {
            $item.Delete()
            Copy-Item -LiteralPath $src -Destination $item.FullName -Force
            $fixed++
        } catch {
            $broken++
        }
    } else {
        $broken++
    }
}

if ($fixed -gt 0) {
    Write-Output "  修复了 $fixed 个失效链接"
} else {
    Write-Output "  链接正常，无需修复"
}
if ($broken -gt 0) {
    Write-Output "  [警告] $broken 个文件没能修复（内置 Python 里找不到对应文件）"
}

exit 0
