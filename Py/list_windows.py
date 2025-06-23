import ctypes
import ctypes.wintypes
import win32gui
import win32con
import win32process
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_all_windows():
    """List all top-level windows with their class names and titles."""
    windows = []
    
    def enum_windows_callback(hwnd, _):
        try:
            # Get window title
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd) + 1
            title = ctypes.create_unicode_buffer(length)
            ctypes.windll.user32.GetWindowTextW(hwnd, title, length)
            title = title.value.strip()
            
            # Get window class name
            class_name = win32gui.GetClassName(hwnd)
            
            # Get process name
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process_name = ""
            try:
                process_handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
                process_name = win32process.GetModuleFileNameEx(process_handle, 0)
                process_name = os.path.basename(process_name)
            except:
                pass
            
            # Get window style
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            
            # Get window rect
            try:
                rect = win32gui.GetWindowRect(hwnd)
                width = rect[2] - rect[0]
                height = rect[3] - rect[1]
                size = f"{width}x{height}"
            except:
                size = "N/A"
            
            # Skip if window is not visible or has no title
            if not title or not ctypes.windll.user32.IsWindowVisible(hwnd):
                return True
                
            windows.append({
                'hwnd': hwnd,
                'title': title,
                'class': class_name,
                'process': process_name,
                'style': style,
                'size': size,
                'visible': bool(ctypes.windll.user32.IsWindowVisible(hwnd)),
                'iconic': bool(win32gui.IsIconic(hwnd))
            })
            
        except Exception as e:
            logger.error(f"Error processing window {hwnd}: {e}")
            
        return True
    
    # Enumerate all top-level windows
    ctypes.windll.user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.POINTER(ctypes.c_int))(enum_windows_callback), 0)
    
    return windows

def main():
    import os
    import win32api
    
    print("Listing all top-level windows...\n")
    windows = list_all_windows()
    
    # Print header
    print(f"{'HWND':<12} {'Class':<30} {'Process':<20} {'Size':<15} {'Visible':<8} {'Title'}")
    print("-" * 100)
    
    # Print each window
    for window in sorted(windows, key=lambda w: (w['process'], w['class'], w['title'])):
        print(f"{window['hwnd']:<12} {window['class']:<30} {window['process']:<20} {window['size']:<15} {window['visible']:<8} {window['title']}")

if __name__ == "__main__":
    main()
