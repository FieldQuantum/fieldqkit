param(
    [string]$Root = (Resolve-Path ".").Path
)

$ErrorActionPreference = "Stop"

# 统计这些扩展名（可按需增减）
$ext = @(
    ".py",".m",".jl",".c",".cpp",".h",".hpp",".cs",".java",
    ".js",".ts",".tsx",".jsx",".go",".rs",".swift",".kt",
    ".scala",".r",".sh",".ps1"
)

# 排除目录
$excludePattern = '\\(\.git|node_modules|venv|\.venv|build|dist|out|\.idea|\.vscode)\\'

Write-Host "Root: $Root"
Write-Host "Scanning files..."

$files = Get-ChildItem -Path $Root -Recurse -File | Where-Object {
    ($ext -contains $_.Extension.ToLower()) -and ($_.FullName -notmatch $excludePattern)
}

$total = 0
$byExt = @{}

foreach ($f in $files) {
    $n = (Get-Content -Path $f.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    $total += $n

    $e = $f.Extension.ToLower()
    if (-not $byExt.ContainsKey($e)) { $byExt[$e] = 0 }
    $byExt[$e] += $n
}

Write-Host ""
Write-Host "========== RESULT =========="
Write-Host ("Files: {0}" -f $files.Count)
Write-Host ("Total lines: {0}" -f $total)
Write-Host ""

$byExt.GetEnumerator() |
    Sort-Object Name |
    Format-Table Name, Value -AutoSize