iptables -t nat -A POSTROUTING -o wlp2s0 -j MASQUERADE
iptables -A FORWARD -i wlp2s0 -o enp0s20f0u2c2 -m state --state RELATED,ESTABLISHED -j ACCEPT
iptables -A FORWARD -i enp0s20f0u2c2 -o wlp2s0 -j ACCEPT
