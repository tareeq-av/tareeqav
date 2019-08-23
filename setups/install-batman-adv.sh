# Install batman-adv
sudo apt-get install -y batctl
# Have batman-adv startup automatically on boot
echo 'batman-adv' | sudo tee --append /etc/modules
# Prevent DHCPCD from automatically configuring wlan0, THIS IS KEY
echo 'denyinterfaces wlan0' | sudo tee --append /etc/dhcpcd.conf
# Enable interfaces on boot
echo "$(pwd)/start-batman-adv.sh" >> ~/.bashrc
# And done!
echo "Reboot to complete install."
