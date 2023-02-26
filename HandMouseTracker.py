import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import concurrent.futures

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

######################## VARIAVEIS #########################

WINDOW_WIDTH = 200
WINDOW_HEIGHT = 150
mouse = Controller()
CURRENT_X, CURRENT_Y = mouse.position
MAX_HANDS = 1
DETECTION_CONFIDENCE = 0.8
TRACKING_CONFIDENCE = 0.8
hands = mp.solutions.hands.Hands(
    max_num_hands=MAX_HANDS,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE)

################################# FUNCOES #####################################
# Espelha a imagem horizontalmente
def horizontal_flip(image):
    return cv2.flip(image, 1)
# Obtém a posicão do dedo indicador direito
def get_index_finger_position(image, hand_landmarks):
    x = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * 1360)
    y = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * 768)
    return x, y
# Move o mouse para a posição especificada
def move_mouse(x, y):
    global CURRENT_X, CURRENT_Y
    num_steps = 2
    x_diff = (x - CURRENT_X) / num_steps
    y_diff = (y - CURRENT_Y) / num_steps
    for i in range(num_steps):
        mouseLoc = (int(CURRENT_X + i * x_diff),
                    int(CURRENT_Y + i * y_diff))
        mouse.position=mouseLoc 
    CURRENT_X = x
    CURRENT_Y = y
# Clica com o botão esquerdo do mouse
def left_click(call, hold_or_not):
    if (call == "DOWN"):
        mouse.press(Button.left)
        mouse.release(Button.left)
    elif (call == "HOLD"):     
        mouse.press(Button.left)
    if (hold_or_not == False):
        mouse.release(Button.left)
# Verifica se houve movimento dos dedos médio e indicador e se o polegar está aberto
def is_finger_moved(hand_landmarks):
    index_finger_tip = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * 768)
    index_finger_dip = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP].y * 768)
    index_finger_pip = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y * 768)
    middle_finger_tip = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y * 768)
    middle_finger_dip = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP].y * 768)
    middle_finger_pip = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y * 768)
    thumb_finger_tip = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x * 1360)
    thumb_finger_mcp = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP].x * 1360)
    thumb_finger_ip = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].x * 1360)
    click_or_not = index_finger_tip <= index_finger_dip <= index_finger_pip and middle_finger_tip >= middle_finger_dip >= middle_finger_pip #Vedadeiro = CLICK, Falso = solta
    hold_or_not = (thumb_finger_tip <= thumb_finger_ip <= thumb_finger_mcp) # Aberto = Verdadeiro, Fechado = Falso
    if click_or_not and not hold_or_not:
        left_click("DOWN", hold_or_not)
    elif click_or_not and hold_or_not:
        left_click("HOLD", hold_or_not)
    elif not click_or_not and not hold_or_not:
        left_click("UP", hold_or_not)

##################### INICIO #################################################
cap = cv2.VideoCapture(0)

# Processa os quadros da câmera paralelamente com threads
with concurrent.futures.ThreadPoolExecutor() as executor:
    while True:
        ret, image = cap.read()
        image = horizontal_flip(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(200,150))
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(22, 66, 121), thickness=1, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(97, 46, 250), thickness=1, circle_radius=2),)                
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label
                if results.multi_handedness[handIndex].classification[0].label == "Right":
                    x, y = get_index_finger_position(image, hand_landmarks)
                    move_mouse(x, y)
                    is_finger_moved(hand_landmarks)
                elif(results.multi_handedness[handIndex].classification[0].label == "Left"):
                    print("Coloque a outra mão")     
        cv2.imshow('Hand Tracking', image)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()