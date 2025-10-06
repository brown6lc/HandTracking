#include <Servo.h>

Servo myServo;
int angle = 90;

void setup() {
  Serial.begin(9600);
  myServo.attach(9);  // Servo on pin 9
  myServo.write(angle);
}

void loop() {
  if (Serial.available() > 0) {
    int newAngle = Serial.parseInt();
    if (newAngle >= 0 && newAngle <= 180) {
      myServo.write(newAngle);
    }
  }
}
