$ErrorActionPreference = 'Stop'

$settings = @{
    Severity = @(
        'Information',
        'Warning',
        'Error'
    )
    IncludeDefaultRules = $true
    ExcludeRules = @(
        'PSAvoidUsingInvokeExpression'
    )
    # Additional rules that are disabled by default.
    #
    # Some of the skips could be replaced with inline comments if PSScriptAnalyzer
    # supports that in the future (https://github.com/PowerShell/PSScriptAnalyzer/issues/849).
    Rules = @{
        PSAvoidExclaimOperator = @{
            Enable = $true
        }
        PSAvoidLongLines = @{
            Enable = $true
            MaximumLineLength = 120
        }
        PSAvoidSemicolonsAsLineTerminators = @{
            Enable = $true
        }
        PSPlaceCloseBrace = @{
            Enable = $true
            NoEmptyLineBefore = $true
            IgnoreOneLineBlock = $true
            NewLineAfter = $false
        }
        PSPlaceOpenBrace = @{
            Enable = $true
            OnSameLine = $true
            NewLineAfter = $true
            IgnoreOneLineBlock = $true
        }
        PSUseConsistentIndentation = @{
            Enable = $true
            IndentationSize = 4
            PipelineIndentation = 'IncreaseIndentationAfterEveryPipeline'
            Kind = 'space'
        }
        PSUseConsistentWhitespace = @{
            Enable = $true
            CheckInnerBrace = $true
            CheckOpenBrace = $true
            CheckOpenParen = $true
            CheckOperator = $true
            CheckSeparator = $true
            CheckPipe = $true
            CheckPipeForRedundantWhitespace = $true
            CheckParameter = $true
            IgnoreAssignmentOperatorInsideHashTable = $false
        }
        PSUseCorrectCasing = @{
            Enable = $true
        }
    }
}

# this pre-listing of files can be removed whenever PSScriptAnalyzer adds support for exclusions.
#
# see:
#
#  * https://github.com/PowerShell/PSScriptAnalyzer/issues/561
#  * https://github.com/PowerShell/vscode-powershell/issues/3048
#
# lint-powershell.ps1 itself is included here because linting this script itself
# sometimes fails (non-deterministically!) with an error like "Object reference not set to an instance of an object"
#
$files = @(
    Get-ChildItem -Path ./ -Recurse -Force -Filter '*.ps1' |
        Where-Object { $_.FullName -notmatch '[/\\]bin[/\\]' } |
        Where-Object { $_.FullName -notmatch '[/\\]external_libs[/\\]' } |
        Where-Object { $_.FullName -notmatch '[/\\]\.pixi[/\\]' } |
        Where-Object { $_.FullName -notmatch '[/\\]venv[/\\]' } |
        Where-Object { $_.Name -ne 'lint-powershell.ps1' } |
        ForEach-Object { $_.FullName }
)

foreach ($file in $files) {
    Write-Output "linting '$file'"
    Invoke-ScriptAnalyzer -Path $file -EnableExit -Settings $settings
}
