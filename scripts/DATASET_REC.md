python scripts/record_dataset.py \
    --img-server-ip 192.168.123.164 \
    --task-dir ./my_datasets/ \
    --task-name pick_red_cube \
    --frequency 30.0 \
    --task-goal "Поднять красный кубик со стола" \
    --task-desc "Демонстрация телеуправления G1" \
    --task-steps "1. Подвести руки. 2. Схватить кубик. 3. Поднять."

Предварительно запустить Image server, сконфигурировав камеры.

https://github.com/unitreerobotics/teleimager/tree/57cf2a40572227273fa001cd17833b755331ec97