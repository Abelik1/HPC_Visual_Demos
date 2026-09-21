# Clean up the desktop for the demo stand, and put it all back afterwards.
#
# Run it once to go into presentation mode, and again to come back out:
#
#   .\scripts\presentation_mode.ps1          # toggles
#   .\scripts\presentation_mode.ps1 -On      # or say which
#   .\scripts\presentation_mode.ps1 -Off
#
# Run it from an ordinary PowerShell window.  The saved settings and your
# wallpaper are kept in %USERPROFILE%\.hpcdemo\presentation; leave it there.
#
# Presentation mode hides the desktop icons, auto-hides the taskbar, sets a
# plain dark background, turns notifications off, and stops the screen and the
# laptop from sleeping.  Then it opens the two dashboard windows in a clean
# Chrome profile of their own: the public viewer (/demo) full screen, the
# presenter dashboard (/) beside it.  It does not start the server.
#
# All of these are computer-wide, not per virtual desktop, so the settings as
# they were are saved on the way in and restored on the way out.  Run it on
# a fresh virtual desktop (Win+Ctrl+D) so no other windows are in the way, and
# pick the display mode (Win+P -> Extend) first: the windows are placed on the
# monitors that exist when it runs.

[CmdletBinding(DefaultParameterSetName = 'Toggle')]
param(
    [Parameter(ParameterSetName = 'On')][switch]$On,
    [Parameter(ParameterSetName = 'Off')][switch]$Off,
    [string]$Url = 'http://127.0.0.1:8000',
    [string]$Background = '20 20 24',
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'

# Not under AppData: when this runs from a packaged app (the Claude desktop app,
# for one) writes there are redirected into the app's private folder, and
# Explorer then cannot find the saved wallpaper to put it back.
$stateDir  = Join-Path $env:USERPROFILE '.hpcdemo\presentation'
$stateFile = Join-Path $stateDir 'saved_settings.json'

Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System;
using System.Runtime.InteropServices;

public static class DemoShell {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int left, top, right, bottom; }

    [StructLayout(LayoutKind.Sequential)]
    public struct APPBARDATA {
        public uint cbSize; public IntPtr hWnd; public uint uCallbackMessage;
        public uint uEdge; public RECT rc; public IntPtr lParam;
    }

    [DllImport("shell32.dll")] static extern UIntPtr SHAppBarMessage(uint msg, ref APPBARDATA data);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern IntPtr FindWindow(string cls, string title);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern IntPtr FindWindowEx(IntPtr parent, IntPtr after, string cls, string title);
    [DllImport("user32.dll")] static extern IntPtr SendMessage(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern bool SystemParametersInfo(uint action, uint param, string value, uint flags);
    [DllImport("user32.dll")] static extern bool SetSysColors(int count, int[] elements, int[] colors);

    const uint ABM_GETSTATE = 4, ABM_SETSTATE = 10, ABS_AUTOHIDE = 1, ABS_ALWAYSONTOP = 2;

    static APPBARDATA Taskbar() {
        var d = new APPBARDATA();
        d.cbSize = (uint)Marshal.SizeOf(d);
        d.hWnd = FindWindow("Shell_TrayWnd", null);
        return d;
    }

    public static bool TaskbarAutoHide {
        get { var d = Taskbar(); return ((uint)SHAppBarMessage(ABM_GETSTATE, ref d) & ABS_AUTOHIDE) != 0; }
        set {
            var d = Taskbar();
            d.lParam = (IntPtr)(value ? ABS_AUTOHIDE : ABS_ALWAYSONTOP);
            SHAppBarMessage(ABM_SETSTATE, ref d);
        }
    }

    // The icons live in a SHELLDLL_DefView under Progman, or under one of the
    // WorkerW windows once a wallpaper slideshow or animation has run.
    static IntPtr DesktopView() {
        IntPtr view = FindWindowEx(FindWindow("Progman", null), IntPtr.Zero, "SHELLDLL_DefView", null);
        IntPtr worker = IntPtr.Zero;
        while (view == IntPtr.Zero) {
            worker = FindWindowEx(IntPtr.Zero, worker, "WorkerW", null);
            if (worker == IntPtr.Zero) break;
            view = FindWindowEx(worker, IntPtr.Zero, "SHELLDLL_DefView", null);
        }
        return view;
    }

    public static bool DesktopIconsVisible {
        get {
            IntPtr list = FindWindowEx(DesktopView(), IntPtr.Zero, "SysListView32", null);
            return list != IntPtr.Zero && IsWindowVisible(list);
        }
        set {
            // The same command as View -> Show desktop icons; it flips the
            // setting, so only send it when the state is not already right.
            if (value != DesktopIconsVisible)
                SendMessage(DesktopView(), 0x0111, (IntPtr)0x7402, IntPtr.Zero);
        }
    }

    public static bool SetWallpaper(string path) {
        return SystemParametersInfo(0x0014, 0, path, 0x01 | 0x02);
    }

    public static void SetBackgroundColor(int r, int g, int b) {
        SetSysColors(1, new[] { 1 }, new[] { r | (g << 8) | (b << 16) });
    }
}
'@

$notifKey  = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Notifications\Settings'
$notifName = 'NOC_GLOBAL_SETTING_TOASTS_ENABLED'
$powerSettings = @(
    @{ Name = 'screen'; Sub = 'SUB_VIDEO'; Setting = 'VIDEOIDLE' },
    @{ Name = 'sleep';  Sub = 'SUB_SLEEP'; Setting = 'STANDBYIDLE' }
)

function Get-PowerTimeouts($sub, $setting) {
    $text = (powercfg /q SCHEME_CURRENT $sub $setting) -join "`n"
    $ac = [regex]::Match($text, 'AC Power Setting Index:\s*0x([0-9a-fA-F]+)')
    $dc = [regex]::Match($text, 'DC Power Setting Index:\s*0x([0-9a-fA-F]+)')
    if (-not ($ac.Success -and $dc.Success)) { throw "Could not read the $setting power setting." }
    @{ AC = [Convert]::ToInt32($ac.Groups[1].Value, 16); DC = [Convert]::ToInt32($dc.Groups[1].Value, 16) }
}

function Set-PowerTimeouts($sub, $setting, $ac, $dc) {
    powercfg /setacvalueindex SCHEME_CURRENT $sub $setting $ac | Out-Null
    powercfg /setdcvalueindex SCHEME_CURRENT $sub $setting $dc | Out-Null
}

function Set-Background($wallpaper, $color) {
    Set-ItemProperty 'HKCU:\Control Panel\Colors' -Name Background -Value $color
    $r, $g, $b = $color -split '\s+' | ForEach-Object { [int]$_ }
    [DemoShell]::SetBackgroundColor($r, $g, $b)
    if (-not [DemoShell]::SetWallpaper($wallpaper)) {
        throw "Windows would not set the wallpaper to '$wallpaper'.  Set it back by hand (Settings > Personalization > Background)."
    }
}

function Find-Chrome {
    $candidates = @(
        "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
    )
    $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

function Open-Window($chrome, $profileName, $address, $screen, [switch]$FullScreen) {
    # Each window gets its own profile directory, so each is its own Chrome
    # process and the position and full-screen flags are honoured (a second
    # window in a running Chrome ignores them).  The profiles are kept between
    # runs: no sign-in, no history, no extensions, just the dashboard.
    $b = $screen.WorkingArea
    $chromeArgs = @(
        "--user-data-dir=`"$(Join-Path $stateDir $profileName)`"",
        '--no-first-run', '--no-default-browser-check', '--hide-crash-restore-bubble',
        "--app=$address",
        "--window-position=$($b.X),$($b.Y)", "--window-size=$($b.Width),$($b.Height)"
    )
    if ($FullScreen) { $chromeArgs += '--start-fullscreen' } else { $chromeArgs += '--start-maximized' }
    Start-Process $chrome -ArgumentList $chromeArgs
}

function Enable-PresentationMode {
    if (Test-Path $stateFile) {
        # Saving again would overwrite the real settings with the presentation ones.
        Write-Host 'Presentation mode is already on; run with -Off to restore.'
        return
    }
    New-Item -ItemType Directory -Force $stateDir | Out-Null

    # Copy the wallpaper out first: Windows keeps it in a file it overwrites
    # as soon as the background changes.
    $wallpaper = (Get-ItemProperty 'HKCU:\Control Panel\Desktop').WallPaper
    $wallpaperCopy = ''
    if ($wallpaper -and (Test-Path $wallpaper)) {
        $ext = [IO.Path]::GetExtension($wallpaper)
        if (-not $ext) { $ext = '.jpg' }   # TranscodedWallpaper has none; it is a JPEG
        $wallpaperCopy = Join-Path $stateDir ('wallpaper' + $ext)
        # After a restore the wallpaper already is that copy.
        if ((Resolve-Path $wallpaper).Path -ne $wallpaperCopy) { Copy-Item $wallpaper $wallpaperCopy -Force }
    }
    $notif = (Get-ItemProperty $notifKey -Name $notifName -ErrorAction SilentlyContinue).$notifName

    $saved = [ordered]@{
        DesktopIcons  = [DemoShell]::DesktopIconsVisible
        TaskbarHide   = [DemoShell]::TaskbarAutoHide
        Wallpaper     = $wallpaperCopy
        Background    = (Get-ItemProperty 'HKCU:\Control Panel\Colors').Background
        Notifications = $notif
        Power         = @{}
    }
    foreach ($p in $powerSettings) { $saved.Power[$p.Name] = Get-PowerTimeouts $p.Sub $p.Setting }
    # Written before anything changes, so a failure half-way can still be undone with -Off.
    $saved | ConvertTo-Json -Depth 4 | Set-Content $stateFile -Encoding utf8

    [DemoShell]::DesktopIconsVisible = $false
    [DemoShell]::TaskbarAutoHide = $true
    Set-Background '' $Background
    Set-ItemProperty $notifKey -Name $notifName -Value 0 -Type DWord
    foreach ($p in $powerSettings) { Set-PowerTimeouts $p.Sub $p.Setting 0 0 }
    powercfg /setactive SCHEME_CURRENT | Out-Null

    Write-Host 'Presentation mode on: icons hidden, taskbar auto-hides, plain background, notifications off, no sleep.'

    if ($NoBrowser) { return }
    $chrome = Find-Chrome
    if (-not $chrome) { Write-Warning 'Chrome not found; open the dashboard yourself.'; return }

    # The public viewer goes on an external monitor when there is one; the
    # presenter dashboard on a second external monitor, or else the laptop.
    $primary  = [System.Windows.Forms.Screen]::PrimaryScreen
    $external = @([System.Windows.Forms.Screen]::AllScreens | Where-Object { -not $_.Primary })
    $publicScreen = if ($external.Count -ge 1) { $external[0] } else { $primary }
    $driverScreen = if ($external.Count -ge 2) { $external[1] } else { $primary }

    $base = $Url.TrimEnd('/')
    Open-Window $chrome 'chrome-presenter' "$base/" $driverScreen
    Open-Window $chrome 'chrome-public' "$base/demo" $publicScreen -FullScreen

    if ($external.Count -eq 0) {
        Write-Host 'Only one display found, so both windows are on it.  Drag the /demo window to the public monitor and press F11.'
    }
}

function Disable-PresentationMode {
    if (-not (Test-Path $stateFile)) { Write-Host 'Presentation mode is not on; nothing to restore.'; return }
    $saved = Get-Content $stateFile -Raw | ConvertFrom-Json

    [DemoShell]::DesktopIconsVisible = [bool]$saved.DesktopIcons
    [DemoShell]::TaskbarAutoHide = [bool]$saved.TaskbarHide
    Set-Background $saved.Wallpaper $saved.Background
    if ($null -eq $saved.Notifications) {
        Remove-ItemProperty $notifKey -Name $notifName -ErrorAction SilentlyContinue
    } else {
        Set-ItemProperty $notifKey -Name $notifName -Value ([int]$saved.Notifications) -Type DWord
    }
    foreach ($p in $powerSettings) {
        $t = $saved.Power.($p.Name)
        Set-PowerTimeouts $p.Sub $p.Setting $t.AC $t.DC
    }
    powercfg /setactive SCHEME_CURRENT | Out-Null

    Remove-Item $stateFile
    Write-Host 'Presentation mode off: settings restored.  The dashboard windows are left open; close them yourself.'
}

$turnOn = if ($On) { $true } elseif ($Off) { $false } else { -not (Test-Path $stateFile) }
if ($turnOn) { Enable-PresentationMode } else { Disable-PresentationMode }
