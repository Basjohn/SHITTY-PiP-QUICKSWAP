# Shitty PiP QuickSwap (SPQ)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)](https://www.microsoft.com/windows/)

Shitty Picture In Picture QuickSwap is a multiple overlay application allowing smart swapping of captured content with your current window and also has a worse monitor overlay for the lulz. And is a genuine alternative to alt-tab that obeys MRU. Features intelligent switching, routed media control and now multioverlay docking is supported with full swap capabilities.

<img width="636" height="347" alt="{7C387F2E-DCC7-4333-A301-2BEC66E6A929}" src="https://github.com/user-attachments/assets/057bbd95-0ba0-4de1-88ab-ffeccfd0dfc3" />




## Features

#### NEWESTER
Docking. Up to 5 simultaneous overlays can be made that sync together with your window usage, each other and allow targetted swaps into applications. Fully size adjustable and orientates to whatever corner you place it in.

### NEWEST
Rebuilt from the ground up with new deep knowledge like files shouldn't import each other, there isn't a fragment of the old code remaining but many more functional features at far greater performance.

For a brief rundown without the details I don't understand myself: Faster, uses less of every resource, reliable aspect ratio control with constant corrections, safe media passthrough keys (you can still enable less safe basic bitch passthrough), smarter quick switching, smarter auto-switching, lock functionality (click the circle) for auto-switching, border customization, A FUCKING SCROLL WHEEL IN THE SUBSETTINGS HOLY SHI-, anyway, hardware accelerated monitor capture, global customizable opacity hotkeys on a gradual hold timer, editable blacklist to prevent your dumbass from using unsafe keypassthrough on known games with anti-cheat. Better media keep alive logic, though some apps will always require a nudge by the very nature of their creation and culling.

Display Locked Switching when enabled will not let autoswitch or quickswitch get you all muddled if you have multiple displays. Ensuring if you start a Display 2 window on Display 1 it only switches with other Display 2 windows.

Media key functionality is especially fun. Spacebar pause/play, left right for backwards and forwards, up and down for windows mixer volume control of the captured application with a snazzy OSD.

Every feature (except old ass keypassthrough) built to standards that should be completely compatible with even the most anal anit-cheats by using legitimate windows api functions.
----------
## OLD SAUCE - EVERYTHING BELOW THIS LINE IS DEPRECATED AND EVEN MORE SILLY.
----------
- * Live Window Overlay creation that tracks your most recent application and let's you seamlessly swap applications like Microsoft remembered what Alt-Tab was meant to do. Swapping can be done via double-click, right click or custom hotkey combo.

- * Overlay Passthrough. This was surprisingly less hellish than everything else to get working but you probably DON'T WANT TO TURN THIS ON if you game online. DEFAULTS TO OFF, gives a big fucking warning too. It's a benign feature but I don't want false positives on users. Neat for pure media though, when it works. You can't accidentally turn this on.

- * Live MONITOR Overlay with seamless swap to other displays, multiple fps speed option. DXGI got unfucked so you've got decent monitor capture at long last! I have still have no bloody idea why you would want this, BUT NOW YOU HAVE IT!
 
- * Super sexy settings menu. YES THIS IS A FEATURE. Designed to look like an app that never should have worked, you have a clean, themed, multiple route interface for whatever the hell you'll use this for. With full feature context menus built into every overlay and the system tray. Subsettings (The Settings *inside* SETTINGS!) give live opacity, hot key, fps control and sorting algorithm. 

#There's also a fairly easy to find easter egg that makes me cling to the days I was not entirely a rotting bag of flesh.

   ```
## Usage

1. Run the application
 
2. The application will start with its settings panel, you can also right-click the tray icon to access the menu. Or right click inside an overlay to access a menu with more menus. 

3. To create a window overlay:
   - Click "Window Mode" in the menu
   - Click on the window you want to capture and then >>

4. To create a monitor overlay:
   - Click "Monitor Mode" in the menu
   - Click on the monitor you want to capture and >>

Default hotkeys
` = Quickswitch
= = Increase Opacity
- = Decrease Opacity

Full customizable or removable.

### Settings

Access the subsettings panel from the system tray, context menu or main window.

- Hotkeys
- Theme preferences
- Performance options
- Keypassthrough
- Media Control when overlay is focused
- Display Locked Switching
- Rounded/Sharp Borders

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
