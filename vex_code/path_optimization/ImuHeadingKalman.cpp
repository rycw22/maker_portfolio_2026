#include "ImuHeadingKalman.h"

// Wrap angle to [-pi, pi]
double ImuHeadingKalman::wrapPi(double a) {
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

ImuHeadingKalman::ImuHeadingKalman() {
  theta = 0.0;
  bias  = 0.0;

  P00 = 1.0; P01 = 0.0;
  P10 = 0.0; P11 = 1.0;

  // Estimated tuning
  Q_theta = 1e-3;
  Q_bias  = 1e-6;
  R_meas  = 5e-3;
}

void ImuHeadingKalman::reset(double theta0_rad) {
  theta = wrapPi(theta0_rad);
  bias  = 0.0;

  P00 = 1.0; P01 = 0.0;
  P10 = 0.0; P11 = 1.0;
}

void ImuHeadingKalman::predict(double gyro_rate_rad_s, double dt) {
  // State prediction
  theta = wrapPi(theta + (gyro_rate_rad_s - bias) * dt);

  // F = [[1, -dt],
  //      [0,  1 ]]
  double F00 = 1.0, F01 = -dt;
  double F10 = 0.0, F11 = 1.0;

  // FP = F * P
  double FP00 = F00 * P00 + F01 * P10;
  double FP01 = F00 * P01 + F01 * P11;
  double FP10 = F10 * P00 + F11 * P10;
  double FP11 = F10 * P01 + F11 * P11;

  // P = FP * F^T + Q
  double newP00 = FP00 * F00 + FP01 * F01;
  double newP01 = FP00 * F10 + FP01 * F11;
  double newP10 = FP10 * F00 + FP11 * F01;
  double newP11 = FP10 * F10 + FP11 * F11;

  P00 = newP00 + Q_theta;
  P01 = newP01;
  P10 = newP10;
  P11 = newP11 + Q_bias;
}

void ImuHeadingKalman::update(double meas_theta_rad) {
  double z = wrapPi(meas_theta_rad);
  double y = wrapPi(z - theta);

  double S = P00 + R_meas;

  double K0 = P00 / S;
  double K1 = P10 / S;

  theta = wrapPi(theta + K0 * y);
  bias  = bias + K1 * y;

  double newP00 = (1.0 - K0) * P00;
  double newP01 = (1.0 - K0) * P01;
  double newP10 = P10 - K1 * P00;
  double newP11 = P11 - K1 * P01;

  P00 = newP00;
  P01 = newP01;
  P10 = newP10;
  P11 = newP11;
}