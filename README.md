# Shitty PiP QuickSwap (SPQ)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)](https://www.microsoft.com/windows/)

Shitty Picture In Picture QuickSwap is a multiple overlay application allowing smart swapping of captured content with your current window and also has a worse monitor overlay for the lulz. And is a genuine alternative to alt-tab that obeys MRU.

<img width="600" alt="FullscreenExample" src="https://github.com/user-attachments/assets/7b833ff5-97d1-4a20-9504-119af4efb4ab" />


## Features

- * Live Window Overlay creation that tracks your most recent application and let's you seamlessly swap applications like Microsoft remembered what Alt-Tab was meant to do. Swapping can be done via double-click, right click or custom hotkey combo.

- * Caveat 1 - Many applications are smart ass bitches and stop animation when you minimize them, DWM cannot avoid this. There are mitigators to try and lie to applications that they are still in focus but it isn't aggressive as basic function was more important. This is not an issue with games unless you use exclusive fullscreen which is a *you* thing. WANT VIDEO?! Only tested solution is something live mpv.net and then NOT minimizing it, just opening your other applications afterwards while it plays. (Maximize others)

- * KINDA SORTA ACTUAL DESKTOP FUCKING CAPTURE. Sure that's a weird thing to be excited about but if you knew how bad pythons capture libraries were and the documentation on DWM saying this is "impossible" well, it ain't. Using the mystical art of reading the god damn window entries the application isolates and lies to you about the desktop using a feature Windows has been meaning to depreciate since W3.x but were too busy fucking up libraries. It could be removed at any time but it's been 30 years, you do the guessing.

- * Overlay Passthrough. This was surprisingly less hellish than everything else to get working but you probably DON'T WANT TO TURN THIS ON if you game online. DEFAULTS TO OFF, gives a big fucking warning too. It's a benign feature but I don't want false positives on users. Neat for pure media though, when it works. You can't accidentally turn this on.

- * Live MONITOR Overlay with seamless swap to and from the Window Overlay, custom fps speed option. (Match your other screen's refresh or just put big numbers and complain about a bsod - I'm kidding, 240 is the limit. People who want more than 240 can send me a motivational letter from their GP.)

- * Caveat 2 - Microsoft have fucked the python ecosystem royally. WGC/WINRT/DXGI/D3DSHOT are all fucked for this task in different ways. So MSS had to be used. It is not particularly flawless or as smooth as I would like, but it is the only option and I made the UI with this feature in mind so fucking hell I'm putting it in there.

- * Super sexy settings menu. YES THIS IS A FEATURE. Designed to look like an app that never should have worked, you have a clean, themed, multiple route interface for whatever the hell you'll use this for. With full feature context menus built into every overlay and the system tray. Subsettings (The Settings *inside* SETTINGS!) give live opacity, hot key, fps control and sorting algorithm. 

#There's also a fairly easy to find easter egg that makes me cling to the days I was not entirely a rotting bag of flesh.

   ```
## Usage

1. Run the application
 
2. The application will start with its settings panel, you can also right-click the tray icon to access the menu. Or right click inside an overlay to access a menu with more menus. 

3. To create a window overlay:
   - Click "New Window Overlay" in the menu
   - Click on the window you want to capture

4. To create a monitor overlay:
   - Click "New Monitor Overlay" in the menu
   - Click on the monitor you want to capture

- `Ctrl+Shift+O`: Toggle overlay visibility
- `Ctrl+Shift+Q`: Quit application
- `Alt+Click` on overlay: Move overlay
- `Right-Click` on overlay: Show context menu
- `I don't know who the hell put those short-cuts there, I just did the custom hotkey shit so uh, good luck.

Now a gaming example from a game that really could have used prettier snow.

<img width="600" alt="GAMINGEXAMPLE" src="https://github.com/user-attachments/assets/165ea7c8-4823-4ed6-9a8d-07038585486d" />



## Customization

### Themes

You can switch between dark and light themes in the settings panel.
They work slightly better after closing and reopening. I did say slightly.

### Settings

Access the settings panel from the system tray menu to configure:
- Default overlay size and position
- Hotkeys
- Theme preferences
- Startup behavior
- Performance options

## Building from Source

### Prerequisites

- Python 3.8+
- pip
- Git
- Windows 10/11 SDK (for building from source)

### Building
0. Skip this step and just use the exe like a human.
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   Skip this too!
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   THE EXE, IT CALLS TO YOU MORTAL.
   python Py/main.py
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request I can ignore because I have no idea what I'm doing here.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
6. Assume I know what 1-5 means and will do something at some stage.

## License

This project is licensed under the MIT License - which means idgaf.

## Acknowledgments

- Built with [PySide6](https://pypi.org/project/PySide6/)
- Uses [MSS](https://github.com/BoboTiG/python-mss) for screen capture
- Inspired by how ludicrously fucked alt-tab is in W11.
