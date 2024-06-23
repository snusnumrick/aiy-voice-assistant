import aiy.voicehat


def led_on(led):
    led.set_state(aiy.voicehat.LED.ON)


def led_off(led):
    led.set_state(aiy.voicehat.LED.OFF)
