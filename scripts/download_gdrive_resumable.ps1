[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$FileId,
    [Parameter(Mandatory = $true)][string]$Destination,
    [Parameter(Mandatory = $true)][long]$ExpectedBytes,
    [int]$MaxAttempts = 20
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http
$Destination = [IO.Path]::GetFullPath($Destination)
$destinationDir = Split-Path -Parent $Destination
New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

function Get-GoogleDriveConfirmedUrl {
    param([string]$Id)

    $warningUrl = "https://drive.google.com/uc?export=download&id=$Id"
    $response = Invoke-WebRequest -Uri $warningUrl -UseBasicParsing -TimeoutSec 120
    $uuid = [regex]::Match($response.Content, 'name="uuid" value="([^"]+)"').Groups[1].Value
    if (!$uuid) {
        throw 'Google Drive confirmation UUID not found; the sharing page may have changed.'
    }
    return "https://drive.usercontent.google.com/download?id=$Id&export=download&confirm=t&uuid=$uuid"
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $current = if (Test-Path -LiteralPath $Destination) {
        (Get-Item -LiteralPath $Destination).Length
    } else {
        0L
    }

    if ($current -eq $ExpectedBytes) {
        Write-Host "verified checkpoint: $Destination ($current bytes)"
        exit 0
    }
    if ($current -gt $ExpectedBytes) {
        throw "Partial file is larger than expected ($current > $ExpectedBytes): $Destination"
    }

    Write-Host "Google Drive attempt $attempt/$MaxAttempts; resuming at byte $current"
    $client = $null
    $response = $null
    $inputStream = $null
    $outputStream = $null
    try {
        $url = Get-GoogleDriveConfirmedUrl -Id $FileId
        $handler = [System.Net.Http.HttpClientHandler]::new()
        $handler.AllowAutoRedirect = $true
        $client = [System.Net.Http.HttpClient]::new($handler)
        $client.Timeout = [TimeSpan]::FromMinutes(20)
        $client.DefaultRequestHeaders.UserAgent.ParseAdd('Mozilla/5.0 VAD-study/1.0')

        $request = [System.Net.Http.HttpRequestMessage]::new(
            [System.Net.Http.HttpMethod]::Get,
            $url
        )
        if ($current -gt 0) {
            $request.Headers.Range = [System.Net.Http.Headers.RangeHeaderValue]::new($current, $null)
        }

        $response = $client.SendAsync(
            $request,
            [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead
        ).GetAwaiter().GetResult()

        $status = [int]$response.StatusCode
        if ($current -gt 0 -and $status -ne 206) {
            throw "Server ignored Range request: HTTP $status. Refusing to append incompatible bytes."
        }
        if ($current -eq 0 -and $status -notin @(200, 206)) {
            throw "Unexpected HTTP status: $status"
        }

        if ($current -gt 0) {
            $contentRange = $response.Content.Headers.ContentRange
            if ($null -eq $contentRange -or $contentRange.From -ne $current) {
                throw "Unexpected Content-Range while resuming at $current"
            }
        }

        $inputStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $mode = if ($current -gt 0) {
            [System.IO.FileMode]::Append
        } else {
            [System.IO.FileMode]::Create
        }
        $outputStream = [System.IO.FileStream]::new(
            $Destination,
            $mode,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None,
            4194304,
            [System.IO.FileOptions]::SequentialScan
        )
        $buffer = [byte[]]::new(4194304)
        while (($read = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $outputStream.Write($buffer, 0, $read)
        }
        $outputStream.Flush($true)
    } catch {
        Write-Warning $_.Exception.Message
    } finally {
        if ($null -ne $outputStream) { $outputStream.Dispose() }
        if ($null -ne $inputStream) { $inputStream.Dispose() }
        if ($null -ne $response) { $response.Dispose() }
        if ($null -ne $client) { $client.Dispose() }
    }

    $after = if (Test-Path -LiteralPath $Destination) {
        (Get-Item -LiteralPath $Destination).Length
    } else {
        0L
    }
    Write-Host "checkpoint now has $after / $ExpectedBytes bytes"
    if ($after -eq $ExpectedBytes) {
        Write-Host "verified checkpoint: $Destination"
        exit 0
    }
    Start-Sleep -Seconds ([Math]::Min(3 * $attempt, 30))
}

$final = if (Test-Path -LiteralPath $Destination) {
    (Get-Item -LiteralPath $Destination).Length
} else {
    0L
}
throw "Google Drive download did not finish after $MaxAttempts attempts ($final / $ExpectedBytes bytes)."
