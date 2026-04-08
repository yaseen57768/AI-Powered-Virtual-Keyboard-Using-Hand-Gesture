import matplotlib
matplotlib.use("Agg")

import cv2
import mediapipe as mp
import numpy as np
import time
import difflib
from pynput.keyboard import Controller
from wordfreq import zipf_frequency, top_n_list

keyboard = Controller()

# ================= SETTINGS =================
languages = ["en","es","fr"]
lang_index = 0

typing_hand = "Right"      # typing hand
gesture_hand = "Left"      # gesture hand

key_size = 65
margin = 30
row_gap = 18
col_gap = 12

tap_cooldown = {}
gesture_cooldown = {}
last_x = {}

# ============================================

WORD_LIST = {l:set(top_n_list(l,3000)) for l in languages}

keys = [
    "ESC 1 2 3 4 5 6 7 8 9 0 - = BACK",
    "TAB Q W E R T Y U I O P [ ] \\",
    "CAPS A S D F G H J K L ; ' ENTER",
    "SHIFT Z X C V B N M , . / SHIFT",
    "CTRL WIN ALT SPACE ALT WIN MENU CTRL"
]

current_word=""
predictions=[]
shift_active=False
caps_active=False

# ================= MEDIAPIPE =================
mp_hands=mp.solutions.hands
hands=mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
draw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

window_width=1300
window_height=800

cv2.namedWindow("AI Keyboard", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Keyboard",window_width,window_height)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,window_height)

# ================= FUNCTIONS =================

def autocorrect(word):
    if not word:return ""
    wordset=WORD_LIST[languages[lang_index]]
    s=difflib.get_close_matches(word,wordset,n=5,cutoff=0.6)
    if not s:return word
    return max(s,key=lambda w:zipf_frequency(w,languages[lang_index]))

def predict(word):
    if not word:return []
    wordset=WORD_LIST[languages[lang_index]]
    return [w for w in wordset if w.startswith(word)][:3]

def detect_sign(hand):
    tips=[8,12,16,20]
    fingers=[hand.landmark[t].y < hand.landmark[t-2].y for t in tips]
    thumb=hand.landmark[4].x < hand.landmark[3].x

    if all(fingers): return "LANG"
    if not any(fingers): return "SPACE"
    if thumb and not any(fingers): return "ENTER"
    if fingers[0] and not any(fingers[1:]): return "SELECT"
    return None

# ============================================

def draw_keyboard(frame):
    pos=[]
    row_y=margin

    for row in keys:
        cols=row.split(" ")
        col_x=margin

        for ch in cols:

            if ch=="SPACE":
                w=key_size*5
            elif ch in ["SHIFT","CAPS","TAB","ENTER","BACK","CTRL","ALT","WIN","MENU"]:
                w=int(key_size*1.6)
            else:
                w=key_size

            h=key_size

            cv2.rectangle(frame,(col_x,row_y),(col_x+w,row_y+h),(40,40,40),-1)
            cv2.rectangle(frame,(col_x,row_y),(col_x+w,row_y+h),(255,255,255),2)

            text=cv2.getTextSize(ch,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)[0]
            tx=col_x+(w-text[0])//2
            ty=row_y+(h+text[1])//2

            cv2.putText(frame,ch,(tx,ty),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

            pos.append((ch,col_x,row_y,w,h))
            col_x+=w+col_gap

        row_y+=key_size+row_gap

    return pos

# ============================================

def check_key(x,y,positions):
    for key,x1,y1,w,h in positions:
        if x1<x<x1+w and y1<y<y1+h:
            return key
    return None

# ============================================

def type_key(key):
    global shift_active,caps_active,current_word

    if key=="SPACE":
        keyboard.press(" "); keyboard.release(" ")
        current_word=""

    elif key=="BACK":
        keyboard.press("\b"); keyboard.release("\b")
        current_word=current_word[:-1]

    elif key=="ENTER":
        keyboard.press("\n"); keyboard.release("\n")
        current_word=""

    elif key=="SHIFT":
        shift_active=not shift_active
        return

    elif key=="CAPS":
        caps_active=not caps_active
        return

    elif key in ["CTRL","ALT","WIN","MENU","TAB"]:
        return

    else:
        char=key.lower()
        if caps_active ^ shift_active:
            char=char.upper()

        keyboard.press(char)
        keyboard.release(char)
        current_word+=char

    if shift_active and key not in ["SHIFT","CAPS"]:
        shift_active=False

# ============================================

while True:

    ret,frame=cap.read()
    if not ret: continue

    frame=cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(rgb)

    positions=draw_keyboard(frame)

    cv2.putText(frame,f"Lang: {languages[lang_index].upper()}",
                (30,window_height-40),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)

    for i,w in enumerate(predictions):
        cv2.putText(frame,w,(40+i*220,window_height-80),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    if result.multi_hand_landmarks:

        for hand_id,(hand,handedness) in enumerate(zip(
                result.multi_hand_landmarks,
                result.multi_handedness)):

            label=handedness.classification[0].label

            draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

            h,w,_=frame.shape
            ix=int(hand.landmark[8].x*w)
            iy=int(hand.landmark[8].y*h)
            tx=int(hand.landmark[4].x*w)
            ty=int(hand.landmark[4].y*h)

            cv2.circle(frame,(ix,iy),10,(0,255,0),-1)

            dist=np.hypot(ix-tx,iy-ty)
            gesture=detect_sign(hand)

            now=time.time()

            # -------- SWIPE DELETE (typing hand only) --------
            if label==typing_hand:
                if hand_id in last_x and ix-last_x[hand_id]>140:
                    keyboard.press('\b'); keyboard.release('\b')
                    current_word=current_word[:-1]
                    predictions=predict(current_word)
                    time.sleep(0.15)

                last_x[hand_id]=ix

            # -------- GESTURE HAND --------
            if label==gesture_hand:

                if gesture=="LANG" and now>gesture_cooldown.get(hand_id,0):
                    lang_index=(lang_index+1)%3
                    gesture_cooldown[hand_id]=now+1

                if gesture=="SPACE" and now>gesture_cooldown.get(hand_id,0):

                    corrected=autocorrect(current_word)

                    if corrected!=current_word:
                        for _ in range(len(current_word)):
                            keyboard.press('\b'); keyboard.release('\b')
                        keyboard.type(corrected)

                    keyboard.press(" "); keyboard.release(" ")
                    current_word=""
                    predictions=[]
                    gesture_cooldown[hand_id]=now+0.7

                if gesture=="ENTER" and now>gesture_cooldown.get(hand_id,0):
                    keyboard.press("\n"); keyboard.release("\n")
                    gesture_cooldown[hand_id]=now+0.7

            # -------- TYPING HAND --------
            if label==typing_hand and (dist<30 or gesture=="SELECT"):

                if now>tap_cooldown.get(hand_id,0):

                    key=check_key(ix,iy,positions)
                    if key:
                        type_key(key)
                        predictions=predict(current_word)
                        tap_cooldown[hand_id]=now+0.35

    cv2.imshow("AI Keyboard",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()