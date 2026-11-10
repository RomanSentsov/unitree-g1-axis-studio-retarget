# Доработки в **feature/hands-dds**

- Входные и выходные данные кистей переведены на радианы по URDF RH56DFTP. Порядок суставов: **[pinky, ring, middle, index, thumb_bend, thumb_rot]**.
- Топики ROS 2 **/hand_state/r** и **/hand_state/l** теперь публикуют радианы.
- В скрипте **scripts/record_dataset.py** **left_ee** и **right_ee** пишутся напрямую в радианах.
- В **unitree_g1/Unitree_g1.py** добавлен контроллер кистей:
   - Добавлен **InspireHandDDS**: топик **rt/inspire/cmd** (**unitree_go.msg.dds_.MotorCmds_**, 12 моторов). Совместим с **xr_teleoperate** и **unitree_sim_isaaclab**.
   - Сохранен **InspireHandTCP**: автоконвертирует радианы в регистры **0..1800**.
- Добавлен сервис-мост **scripts/dftp_inspire_service.py**: слушает топик DDS **rt/inspire/cmd** и перенаправляет команды в Ethernet-кисти RH56DFTP по Modbus TCP (порты 6000).

## Запуск тестов

### 1. Симуляция (Isaac Lab)
Канал DDS 1, топики **rt/lowcmd** и **rt/inspire/cmd**:
```bash
python scripts/teleop_ros2.py --sim
```

### 2. Реальный робот через нативный Unitree DDS (схема xr-teleoperate)
В отдельном терминале (или в фоне) запускается сервис кистей DFTP:
```bash
python scripts/dftp_inspire_service.py
```
После чего телеоперация работает строго через DDS:
```bash
python scripts/teleop_ros2.py --hand-interface dds
```

### 3. Реальный робот через прямой Modbus TCP (без фонового сервиса)
Канал DDS 0 для рук (**rt/arm_sdk**), сокеты TCP на **192.168.123.210/211:6000** для кистей внутри самого процесса телеопа:
```bash
python scripts/teleop_ros2.py --hand-interface tcp
```
