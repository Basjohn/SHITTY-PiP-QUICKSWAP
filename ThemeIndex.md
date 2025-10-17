# SPQ Modular Theme Documentation

This document provides a comprehensive guide to the theming system used in the SPQ Modular application. It documents the purpose and usage of each QSS selector and style definition.

## Table of Contents

1. [Base Styles](#base-styles)
2. [Main Window](#main-window)
3. [Title Bar](#title-bar)
4. [Buttons](#buttons)
   - [Standard Buttons](#standard-buttons)
   - [Special Buttons](#special-buttons)
   - [QSmolselect Button](#qsmolselect-button)
5. [Combo Boxes](#combo-boxes)
6. [Context Menus](#context-menus)
7. [Form Elements](#form-elements)
8. [Labels](#labels)
9. [Overlays](#overlays)

## Base Styles

- `QMainWindow`: Base styling for the main application window
- `QMainWindow::separator`: Handles window splitter styling

## Main Window

- `QFrame#main_frame`: The main content area with semi-transparent background
  - Contains all the main UI elements
  - Has rounded corners matching the application style

## Title Bar

- `SettingsPanel QFrame#titleBar`: The top bar containing window controls
- `#titleIcon`: The application icon displayed in the title bar
  - `qproperty-iconSize`: Controls the size of the icon in pixels (default: 32)
  - Can be adjusted in both dark.qss and light.qss theme files
- `#titleLabel`: The window title text
- `#closeButton`: The window close button (X)
  - Hover and pressed states have special styling

## Buttons

### Standard Buttons
- `QPushButton`: Base button style
  - `:hover`: Hover state styling
  - `:pressed`: Pressed state styling
  - `:disabled`: Disabled state styling

### Special Buttons
- `#startButton`: Primary action button
- `#settingsButton`: Settings/gear button
- `#aboutButton`: Information/help button

### QSmolselect Button
- `QPushButton#QSmolselect`: Compact selection button
  - Fixed height with flexible width (100px - 200px)
  - White border with rounded corners
  - Dark background that turns black on hover
  - Used for mode selection
  - Toggleable state (checked/unchecked)
  - Used in pairs for exclusive selection (WINDOW MODE / MONITOR MODE)

## Combo Boxes

- `QComboBox`: Dropdown selectors
  - Custom dropdown arrow
  - Hover and pressed states
  - Dropdown menu styling

## Context Menus

- `QMenu`: Popup context menus
  - `::item`: Individual menu items
  - `::item:selected`: Hovered menu item
  - `::item:disabled`: Disabled menu items
  - `::separator`: Menu dividers

## Form Elements

- `QLineEdit`: Text input fields
  - Custom background and border
  - Selection highlighting
  - Placeholder text styling

- `QGroupBox`: Group containers
  - Title and border styling
  - Used to group related controls

## Labels

- `QLabel`: Standard text labels
  - `:disabled`: Disabled state
  - `#errorLabel`: Error message styling
  - `#badgeLabel`: Notification badges

## Overlays

- `#loadingOverlay`: Semi-transparent overlay for loading states
  - Used to block interaction during operations

## Theme Files

- `dark.qss`: Dark theme styling
- `light.qss`: Light theme styling

## Best Practices

1. Always use the color picker comments for colors to maintain consistency
2. Keep related styles grouped together
3. Use semantic class names that describe the purpose, not the appearance
4. Document any non-obvious styling decisions
5. Test all interactive states (hover, pressed, disabled)

## Adding New Styles

1. Add the new style definition in both theme files
2. Include color picker comments for all color values
3. Document the new style in this file
4. Test in both light and dark themes
5. Ensure proper contrast and accessibility
