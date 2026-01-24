### /etc/nsswitch.conf -> nakon ovoga mi je mDNS radio
-hosts:          files mdns4_minimal [NOTFOUND=return] dns
+hosts:          files dns mdns4_minimal

### %userprofile%/.wslconfig
[wsl2]
dnsTunneling=false

### Treba skinit usbipd kako bi se USB-u pristupilo
### U windowsu - Potrazi port na kojem je spojen, trebalo bi bit nesto ovako:
usbipd list
BUSID  VID:PID    DEVICE                   STATE
1-2    0403:6015  USB Serial Converter     Shared

### onda ga treba attachat na wsl
usbipd attach --wsl --busid 1-2 --auto-attach

### Onda u WSL-u da bi pronasao port:
ls -l /dev/ttyUSB* /dev/serial/by-id/* 2>/dev/null
lrwxrwxrwx 1 root root        13 Jan 24 17:32 /dev/serial/by-id/usb-FTDI_FT231X_USB_UART_D30AY52N-if00-port0 -> ../../ttyUSB0

### za pokretanje sa usb-om kad na wsl-u
docker run --rm -it \
  --device=/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_D30AY52N-if00-port0:/dev/ttyUSB0 \
  xbee_gnn_img