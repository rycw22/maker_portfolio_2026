#include "control/LTVUnicycleController.hpp"
#include <algorithm>
#include <cmath>

LTVUnicycleController::LTVUnicycleController(double dt, const Mat3& Q, const Mat2& R,
                                             int dareIterations)
  : m_dt(dt), m_Q(Q), m_R(R), m_iters(dareIterations) {}

void LTVUnicycleController::SetMaxOutput(double maxV, double maxOmega) {
  m_maxV = std::abs(maxV);
  m_maxOmega = std::abs(maxOmega);
}

Vec3 LTVUnicycleController::ComputeRobotFrameError(const Pose2d& current,
                                                   const Pose2d& reference) const {
  // Transform field error into robot frame (standard unicycle tracking)
  const double dx = reference.x - current.x;
  const double dy = reference.y - current.y;

  const double c = std::cos(current.theta);
  const double s = std::sin(current.theta);

  Vec3 e;
  e.v[0] =  c*dx + s*dy;                   // ex
  e.v[1] = -s*dx + c*dy;                   // ey
  e.v[2] = wrapAngleRad(reference.theta - current.theta); // eθ
  return e;
}

void LTVUnicycleController::MakeDiscreteAB(double omegaRef, double vRef, Mat3& Ad, Mat32& Bd) const {
  // Continuous-time A,B for error dynamics with inputs as corrections u=[v_corr, omega_corr]
  //
  // ex_dot =  omegaRef * ey - v_corr
  // ey_dot = -omegaRef * ex + vRef * eθ
  // eθ_dot = -omega_corr
  //
  // Ac = [ 0      omega  0
  //       -omega  0      vRef
  //        0      0      0 ]
  // Bc = [ -1  0
  //         0  0
  //         0 -1 ]
  //
  // Discretize with forward Euler: Ad = I + Ac*dt, Bd = Bc*dt

  Mat3 Ac{};
  Ac.a[0][1] = omegaRef;
  Ac.a[1][0] = -omegaRef;
  Ac.a[1][2] = vRef;

  Mat32 Bc{};
  Bc.a[0][0] = -1.0;  Bc.a[0][1] = 0.0;
  Bc.a[1][0] =  0.0;  Bc.a[1][1] = 0.0;
  Bc.a[2][0] =  0.0;  Bc.a[2][1] = -1.0;

  Ad = Mat3::Identity();
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) Ad.a[i][j] += Ac.a[i][j]*m_dt;

  for (int i=0;i<3;i++) for (int j=0;j<2;j++) Bd.a[i][j] = Bc.a[i][j]*m_dt;
}

Mat3 LTVUnicycleController::SolveDARE(const Mat3& A, const Mat32& B) const {
  // Iterative solution:
  // P_{k+1} = Aᵀ P A - Aᵀ P B (R + Bᵀ P B)^{-1} Bᵀ P A + Q
  Mat3 P = m_Q;

  const Mat3 At = transpose(A);
  const Mat23 Bt = transpose(B);

  for (int it=0; it<m_iters; it++) {
    const Mat3 PA  = mul(P, A);
    const Mat2 BtPB = mul(Bt, mul(P, B));
    const Mat2 S = add(m_R, BtPB);
    const Mat2 Sinv = inv2(S);

    const Mat23 BtPA = mul(Bt, PA);
    // Compute Aᵀ P B (S^-1) Bᵀ P A:
    // First: P B (3x2), then Aᵀ (3x3) * (3x2) -> (3x2)
    const Mat32 PB = mul(P, B);
    const Mat32 AtPB = mul(At, PB); // (3x2)

    // middle = (S^-1) * (Bᵀ P A)  => (2x2)*(2x3) = (2x3)
    Mat23 middle{};
    for (int i=0;i<2;i++) for (int j=0;j<3;j++) {
      middle.a[i][j] = Sinv.a[i][0]*BtPA.a[0][j] + Sinv.a[i][1]*BtPA.a[1][j];
    }

    // term = (Aᵀ P B) * middle => (3x2)*(2x3) = (3x3)
    Mat3 term{};
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
      term.a[i][j] = AtPB.a[i][0]*middle.a[0][j] + AtPB.a[i][1]*middle.a[1][j];
    }

    const Mat3 AtPA = mul(At, PA);
    P = add(sub(AtPA, term), m_Q);
  }
  return P;
}

ChassisSpeeds LTVUnicycleController::Calculate(const Pose2d& current,
                                               const Pose2d& reference,
                                               const ChassisSpeeds& refSpeeds) {
  // 1) compute error in robot frame
  const Vec3 e = ComputeRobotFrameError(current, reference);

  // 2) build discrete A,B around the reference (LTV)
  Mat3 Ad{};
  Mat32 Bd{};
  MakeDiscreteAB(refSpeeds.omega, refSpeeds.v, Ad, Bd);

  // 3) solve DARE -> P -> K
  const Mat3 P = SolveDARE(Ad, Bd);
  const Mat23 Bt = transpose(Bd);

  const Mat2 S = add(m_R, mul(Bt, mul(P, Bd)));
  const Mat2 Sinv = inv2(S);

  const Mat23 BtPAd = mul(Bt, mul(P, Ad)); // 2x3
  // K = (R + Bᵀ P B)^-1 (Bᵀ P A)
  Mat23 K{};
  for (int i=0;i<2;i++) for (int j=0;j<3;j++) {
    K.a[i][j] = Sinv.a[i][0]*BtPAd.a[0][j] + Sinv.a[i][1]*BtPAd.a[1][j];
  }

  // 4) u = -K e
  const Vec2 u = mul(K, e);
  double v_corr = -u.v[0];
  double w_corr = -u.v[1];

  // 5) add feedforward, clamp
  ChassisSpeeds out;
  out.v = refSpeeds.v + v_corr;
  out.omega = refSpeeds.omega + w_corr;

  out.v = std::clamp(out.v, -m_maxV, m_maxV);
  out.omega = std::clamp(out.omega, -m_maxOmega, m_maxOmega);
  return out;
}