#ifndef IMU_HEADING_KALMAN_H
#define IMU_HEADING_KALMAN_H

#include <cmath>

class ImuHeadingKalman {
public:
  // state
  double theta; 
  double bias; 

  // covariance matrix
  double P00, P01;
  double P10, P11;

  // noise
  double Q_theta; // noise in process
  double Q_bias;  // bias in the process
  double R_meas;  // noise in measurement

  ImuHeadingKalman();

  void reset(double theta0_rad);
  void predict(double gyro_rate_rad_s, double dt);
  void update(double meas_theta_rad);

private:
  double wrapPi(double angle);
};

#endif