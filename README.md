# python software for reading multimeter display
## dependencies
#### pytesseract, opencv, numpy
## dependency installation
### linux

#### arch linux
```bash
sudo pacman -S python-pip
```

#### debian/ubuntu
```bash
sudo apt install python3-pip
```
#### fedora
```bash
sudo dnf install -y python3-pip
```

### windows
```cmd
python -m ensurepip --upgrade
```

### macOS
```bash
python3 -m ensurepip
```

### installation
```bash
pip install opencv-python pytesseract numpy
```

## start
```bash
python3 OCR.py -h # help


# result

# [ WARN:0@0.861] global cap_gstreamer.cpp:1777 open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1
# -h   --help                      displays this output
# -w   --without_units             displays results even without units (default)
# -u   --unit                      displays result only with readable units
# -c   --camera <camera number>    choosing camera (default 0)
```
