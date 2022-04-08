from delta_kinematics import SimulatedDeltaBot
from signal import pause
from buildhat import Motor
import time
# import keyboard
import sys


# bot = SimulatedDeltaBot(servo_link_length = 85.0, parallel_link_length = 210.0,
#                         servo_displacement = 72, effector_displacement = 20)
# # little guy
#     bot = SimulatedDeltaBot(servo_link_length = 65.0, parallel_link_length = 120.0,
#                             servo_displacement = 50, effector_displacement = 25)

# big guy
bot = SimulatedDeltaBot(servo_link_length=120.0, parallel_link_length=230.0,
                        servo_displacement=60.0, effector_displacement=40.0)

Contract = 'anticlockwise'
Extend = 'clockwise'
Shortest = 'shortest'
CW = 'clockwise'
ACW = 'anticlockwise'

isDryRun = False
armMotor1 = Motor('A')
armMotor2 = Motor('B')
armMotor3 = Motor('C')
gripperMotor = Motor('D')


def handle_motor(speed, pos, apos):
    return 0
    # print("Motor speed:{0} pos:{1} apos:{2}".format(speed, pos, apos))


def init():
    armMotor1.when_rotated = handle_motor
    armMotor2.when_rotated = handle_motor
    armMotor3.when_rotated = handle_motor
    armMotor1.set_default_speed(20)
    armMotor2.set_default_speed(20)
    armMotor3.set_default_speed(20)
    # We don't want to coast after running
    armMotor1._release = False
    armMotor2._release = False
    armMotor3._release = False

    gripperMotor.plimit(1)
    gripperMotor.set_default_speed(100)
    go_home_slow()


def go_home_slow():
    print("Go home slow")
    armMotor1.run_to_position(0, 5, False)
    armMotor2.run_to_position(0, 5, False)
    armMotor3.run_to_position(0, 5, False)
    while (abs(armMotor1.get_aposition()) > 20):
        print("motor1:", armMotor1.get_position(), armMotor1.get_aposition())
        time.sleep(.1)
    time.sleep(1)


def is_safe(theta):
    return theta > -80 and theta < 170


def is_servo_safe(servos):
    theta0, theta1, theta2 = servos
    return is_safe(theta0) and is_safe(theta1) and is_safe(theta2)


def get_direction(lastServo, servo):
    return Shortest
    if(abs(servo-lastServo) > 5):
        return Shortest
    return Contract if servo < lastServo else Extend

# time.sleep(3)


def goto_pos(x0, y0, z0, speed=50, blocking=False, isDryRun=False):
    preScaledServos = bot.reverse(x0, y0, z0)
    # scale -x3 due to gearing
    servos = tuple([x*-3 for x in preScaledServos])

    if(servos == (0, 0, 0)):
        print('Abort: Servo POS is BAD:', servos)
        return
    if(is_servo_safe(servos)):
        theta0, theta1, theta2 = servos

        # print ('Going to:', servos)
        # print ('Motor1:', get_direction(lastTheta0,theta0), theta0)
        # print ('Motor2:', get_direction(lastTheta1,theta1),theta1)
        # print ('Motor3:', get_direction(lastTheta2,theta2), theta2)
        if (not isDryRun):
            # Need a btter way to know when this is done.  We want to run them concurrently, and continue when last is done
            # Here we really need to wait for the slowest one
            armMotor1.run_to_position(theta0, speed, False, Shortest)
            armMotor2.run_to_position(theta1, speed, False, Shortest)
            armMotor3.run_to_position(theta2, speed, blocking, Shortest)
        return servos

    print('ERROR: Servo is BAD, skipping:', servos)
    return servos


def do_square(z0, extent, servos):
    for x0, y0 in ((extent, extent), (-extent, extent), (-extent, -extent), (extent, -extent)):
        print('>>>   square:', x0, y0, z0)
        servos = goto_pos(x0, y0, z0, 50, False, isDryRun)
        if not isDryRun:
            time.sleep(2)
    return servos


def close_hand():
    gripperMotor.run_for_seconds(3.5, speed=100)


def open_hand():
    gripperMotor.run_for_seconds(3.5, speed=-100)


def main():
    init()
    servos = (0, 0, 0)

    isDryRun = False

    # time.sleep(1)
    servos = (0, 0, 0)

    # openHand()
    # time.sleep(1)
    # closeHand()

    # servos = gotoPos(0,0,160,20, True)

    # openHand()

    # time.sleep(1)
    # servos = gotoPos(0,0,260,20, True)

    servos = goto_pos(0, 0, 280, 20, True)
    time.sleep(1)

    close_hand()
    time.sleep(1)

    servos = goto_pos(0, 0, 200, 20, True)
    time.sleep(1)
    open_hand()

    servos = goto_pos(0, 0, 150, 5, True)
    time.sleep(1)

    i = 3
    while (i > 0):
        time.sleep(1)
        servos = goto_pos(0, -70, 180, 30, True)
        time.sleep(1)
        servos = goto_pos(0, 70, 180, 30, True)
        i = i - 1

    time.sleep(1)
    go_home_slow()
    time.sleep(5)


if __name__ == '__main__':
    main()
