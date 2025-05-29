pkgname=multimeter-display-reader
pkgver="0.1.0"
pkgrel="1"
pkgdesc="Multimeter display reading program in Python"
arch=('x86_64')
url="git@github.com:networkjarwy1/reading-multimeter-display"
depends=('python>=3.6' 'python-pytesseract' 'python-opencv' 'python-numpy')
makedepends=('git')
source=(OCR.py)
sha256sums=('SKIP')

build() {
  cd "$srcdir/$pkgname"
  # No build process needed
}

package() {
  cd "$srcdir/$pkgname"

  # Create destination directory
  install -Dm755 OCR.py "$pkgdir/usr/bin/multimeter-ocr"

  # Ensure it runs with Python by inserting a shebang
  sed -i '1i#!/usr/bin/env python3' "$pkgdir/usr/bin/multimeter-ocr"
  chmod +x "$pkgdir/usr/bin/multimeter-ocr"
}
