#Front Right Camera
iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
iptables -A FORWARD -i wlan0 -o usb0 -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i usb0 -o wlan0 -j ACCEPT

#Front Left Camera
iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
iptables -A FORWARD -i wlan0 -o usb1 -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i usb1 -o wlan0 -j ACCEPT
