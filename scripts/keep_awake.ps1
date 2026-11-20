# Keeps Windows from idle-sleeping while this process runs (no settings are changed).
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED; released automatically when the process exits.
Add-Type -Name Power -Namespace Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[Win32.Power]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null
while ($true) { Start-Sleep -Seconds 60 }
