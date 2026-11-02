#pragma once
#include "control/Mat.hpp"

struct Pose2d {
  double x;     // units in inches
  double y;     // units in inches
  double theta; // units in radians
};

struct ChassisSpeeds {
  double v;     // linear velocity in inch/sec
  double omega; // angular velocity in rad/sec
};

class LTVUnicycleController {
public:
  // Q: state error weights (3x3 diagonal)
  // R: input weights (2x2 diagonal)
  LTVUnicycleController(double dt, const Mat3& Q, const Mat2& R,
                        int dareIterations = 60);

  // Returns commanded chassis speeds (v, omega)
  ChassisSpeeds Calculate(const Pose2d& current,
                          const Pose2d& reference,
                          const ChassisSpeeds& refSpeeds);

  void SetMaxOutput(double maxV, double maxOmega);

private:
  double m_dt;
  Mat3 m_Q;
  Mat2 m_R;
  int m_iters;

  double m_maxV = 1e9;
  double m_maxOmega = 1e9;

  // Solve Discrete Algebraic Riccati Equation by fixed-point iteration.
  Mat3 SolveDARE(const Mat3& A, const Mat32& B) const;

  // Build LTV discrete system for error dynamics.
  void MakeDiscreteAB(double omegaRef, double vRef, Mat3& Ad, Mat32& Bd) const;

  // Robot-frame error: [ex, ey, eθ]
  Vec3 ComputeRobotFrameError(const Pose2d& current, const Pose2d& reference) const;
};