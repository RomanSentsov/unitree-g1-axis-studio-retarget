python scripts/record_dataset.py \
    --img-server-ip 192.168.123.164 \
    --task-dir ./my_datasets/ \
    --task-name pick_red_cube \
    --frequency 30.0 \
    --task-goal "Grab an SKU and put it in the box" \
    --task-desc "Test dataset, single camera" \
    --task-steps "1. Grasp SKU 2. Lift it 3. Put it in the box"

Предварительно запустить Image server, сконфигурировав камеры.

https://github.com/unitreerobotics/teleimager/tree/57cf2a40572227273fa001cd17833b755331ec97

