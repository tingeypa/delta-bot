from delta_bigguy import init, goto_pos, open_hand, close_hand, go_home_slow
import time
from queue import Queue, Empty
from threading import Event

CLOSE_HAND_WORK =  'close hand'
TOGGLE_PICKUP_ENABLED =  'toggle pickup enabled'
OPEN_HAND_WORK =  'open hand'
GOTO_REF_DROP_WORK =  'goto ref drop'
GOTO_REF_PICKUP_WORK =  'goto ref pickup'
GOTO_REF_READY_WORK =  'goto ref ready'
PICKUP_WORK =  'pickup'

def main():
    init()
        
    #time.sleep(1)

    # openHand()
    # time.sleep(1)
    # closeHand()

    # gotoPos(0,0,160,20, True)

    # openHand()

    # time.sleep(1)
    # gotoPos(0,0,260,20, True)

    goto_pos(0,0,280,20, True)
    time.sleep(1)

    close_hand()
    time.sleep(1)

    goto_pos(0,0,200,20, True)
    time.sleep(1)
    open_hand()

    goto_pos(0,0,150,5, True)
    time.sleep(1)

    i = 3
    while (i > 0):
        time.sleep(1)
        goto_pos(0,-70,180,30, True)
        time.sleep(1)
        goto_pos(0,70,180,30, True)
        i = i - 1


    time.sleep(1)
    go_home_slow()
    time.sleep(5)

TRACK_WIDTH = 130
TRACK_CENTER = -20

# height => up 170-225 down
CARRY_HEIGHT = 170
DROP_HEIGHT = 200
PICKUP_HEIGHT = 225

PICKUP_ZONE = 0  
DROP_ZONE = 0
DROP_TRACK_POS = 50

FAST = 40
SLOW = 20

TRAVEL_SECONDS = 22
pickup_enabled = False

def process_work_queue(work_que:Queue, abort_work:Event):
    init()
    while True:
        try:
            if (abort_work.is_set()):
                break

            # Get some data
            work = work_que.get(block=True, timeout=3)

            # Process the data
            print('Q << Got work from queue:' + str(work))
            work_type, item_x, item_width, timestamp, in_progress_evt, completed_evt = work

            in_progress_evt.set()

            if work_type == OPEN_HAND_WORK:
                open_hand()
                time.sleep(0.2)

                # Indicate completion
                completed_evt.set()

            if work_type == CLOSE_HAND_WORK:
                close_hand()
                time.sleep(0.2)

                # Indicate completion
                completed_evt.set()

            if work_type == GOTO_REF_DROP_WORK:                
                goto_pos(DROP_TRACK_POS, PICKUP_ZONE, CARRY_HEIGHT, FAST, True)
                time.sleep(0.2)
                goto_pos(DROP_TRACK_POS, PICKUP_ZONE, DROP_HEIGHT, SLOW, True)
                time.sleep(0.2)

                # Indicate completion
                completed_evt.set()

            if work_type == GOTO_REF_PICKUP_WORK:
                ref_pos = TRACK_CENTER + (TRACK_WIDTH * -0.5)
                goto_pos(ref_pos, PICKUP_ZONE, CARRY_HEIGHT, FAST, True)
                time.sleep(0.2)
                goto_pos(ref_pos, PICKUP_ZONE, PICKUP_HEIGHT, SLOW, True)
                time.sleep(0.2)

                # # Indicate completion
                completed_evt.set()

            if work_type == GOTO_REF_READY_WORK:
                ref_pos = TRACK_CENTER + (TRACK_WIDTH * -0.5)
                goto_pos(ref_pos, PICKUP_ZONE, CARRY_HEIGHT, FAST, True)
                time.sleep(0.2)

                # # Indicate completion
                completed_evt.set()
            elif work_type == TOGGLE_PICKUP_ENABLED:
                global pickup_enabled
                pickup_enabled = not pickup_enabled
                print('Pickup Enabled: {0}'.format(pickup_enabled))

            elif work_type == PICKUP_WORK:
                if not pickup_enabled:
                    print('IGNORING pickup - press `e` to toggle')
                else:
                    item_mid = item_x + item_width/2
                
                    track_pos = TRACK_CENTER + (TRACK_WIDTH * -item_mid)
                    
                    # move above the item
                    print('Moving to pickup:{}'.format(track_pos))
                    goto_pos(track_pos, PICKUP_ZONE, CARRY_HEIGHT, FAST,True)

                    #wait for it to arrive
                    time_since_detection = time.time() - timestamp
                    if time_since_detection < TRAVEL_SECONDS:
                    
                        wait_time = TRAVEL_SECONDS-time_since_detection
                        print('---- Waiting for pickup: {0:>.3} seconds'.format(wait_time))
                        time.sleep(wait_time)

                        print('Lowering to pickup:{}'.format(track_pos))
                        goto_pos(track_pos, PICKUP_ZONE, PICKUP_HEIGHT, SLOW, True)
                        time.sleep(0.2)

                        print('Closing')
                        close_hand()
                        time.sleep(0.2)

                        print('Lifting')
                        goto_pos(track_pos, PICKUP_ZONE, CARRY_HEIGHT, SLOW, True)
                        time.sleep(0.2)

                        print('Moving to drop')
                        goto_pos(DROP_TRACK_POS, DROP_ZONE, CARRY_HEIGHT, FAST, True)
                        time.sleep(0.2)

                        print('Lowering to drop')
                        goto_pos(DROP_TRACK_POS, DROP_ZONE, DROP_HEIGHT, SLOW, True)
                        time.sleep(0.2)

                        print('Dropping')
                        open_hand()
                        time.sleep(0.1)

                    else:
                        print('Missed that one')
                # Indicate completion
                completed_evt.set()

        except Empty:
            message = 'Nothing to do - timeout'
            # print(message)

    print('Worker thread signing off!')


if __name__ == '__main__':
    main()
