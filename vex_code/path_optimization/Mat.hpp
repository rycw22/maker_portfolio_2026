#pragma once
#include <cmath>
#include <cstdint>

// set up vectors (OOOHH YEAHHHH)
struct Mat3 {
  double a[3][3]{};

  static Mat3 Identity() {
    Mat3 m;
    m.a[0][0] = 1; m.a[1][1] = 1; m.a[2][2] = 1;
    return m;
  }
};

struct Mat2 {
  double a[2][2]{};

  static Mat2 Identity() {
    Mat2 m;
    m.a[0][0] = 1; m.a[1][1] = 1;
    return m;
  }
};

struct Mat32 { // 3x2
  double a[3][2]{};
};

struct Mat23 { // 2x3
  double a[2][3]{};
};

struct Vec3 {
  double v[3]{};
};

struct Vec2 {
  double v[2]{};
};

// set up important funcs for ltv controller
inline Mat3 add(const Mat3& x, const Mat3& y) {
  Mat3 r;
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) r.a[i][j] = x.a[i][j] + y.a[i][j];
  return r;
}
inline Mat3 sub(const Mat3& x, const Mat3& y) {
  Mat3 r;
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) r.a[i][j] = x.a[i][j] - y.a[i][j];
  return r;
}
inline Mat2 add(const Mat2& x, const Mat2& y) {
  Mat2 r;
  for (int i=0;i<2;i++) for (int j=0;j<2;j++) r.a[i][j] = x.a[i][j] + y.a[i][j];
  return r;
}
inline Mat2 sub(const Mat2& x, const Mat2& y) {
  Mat2 r;
  for (int i=0;i<2;i++) for (int j=0;j<2;j++) r.a[i][j] = x.a[i][j] - y.a[i][j];
  return r;
}

inline Mat3 mul(const Mat3& x, const Mat3& y) {
  Mat3 r;
  for (int i=0;i<3;i++) {
    for (int j=0;j<3;j++) {
      double s = 0;
      for (int k=0;k<3;k++) s += x.a[i][k]*y.a[k][j];
      r.a[i][j] = s;
    }
  }
  return r;
}

inline Mat32 mul(const Mat3& x, const Mat32& y) {
  Mat32 r;
  for (int i=0;i<3;i++) {
    for (int j=0;j<2;j++) {
      double s = 0;
      for (int k=0;k<3;k++) s += x.a[i][k]*y.a[k][j];
      r.a[i][j] = s;
    }
  }
  return r;
}

inline Mat23 mul(const Mat23& x, const Mat3& y) {
  Mat23 r;
  for (int i=0;i<2;i++) {
    for (int j=0;j<3;j++) {
      double s = 0;
      for (int k=0;k<3;k++) s += x.a[i][k]*y.a[k][j];
      r.a[i][j] = s;
    }
  }
  return r;
}

inline Mat2 mul(const Mat23& x, const Mat32& y) {
  Mat2 r;
  for (int i=0;i<2;i++) {
    for (int j=0;j<2;j++) {
      double s = 0;
      for (int k=0;k<3;k++) s += x.a[i][k]*y.a[k][j];
      r.a[i][j] = s;
    }
  }
  return r;
}

inline Mat23 transpose(const Mat32& x) {
  Mat23 r;
  for (int i=0;i<3;i++) for (int j=0;j<2;j++) r.a[j][i] = x.a[i][j];
  return r;
}
inline Mat3 transpose(const Mat3& x) {
  Mat3 r;
  for (int i=0;i<3;i++) for (int j=0;j<3;j++) r.a[i][j] = x.a[j][i];
  return r;
}

inline Vec3 mul(const Mat3& x, const Vec3& y) {
  Vec3 r;
  for (int i=0;i<3;i++) {
    r.v[i] = x.a[i][0]*y.v[0] + x.a[i][1]*y.v[1] + x.a[i][2]*y.v[2];
  }
  return r;
}

inline Vec2 mul(const Mat23& x, const Vec3& y) {
  Vec2 r;
  for (int i=0;i<2;i++) {
    r.v[i] = x.a[i][0]*y.v[0] + x.a[i][1]*y.v[1] + x.a[i][2]*y.v[2];
  }
  return r;
}

inline Mat2 inv2(const Mat2& m) {
  // Inverse of 2x2
  const double a = m.a[0][0], b = m.a[0][1];
  const double c = m.a[1][0], d = m.a[1][1];
  const double det = a*d - b*c;
  Mat2 r;
  r.a[0][0] =  d/det;
  r.a[0][1] = -b/det;
  r.a[1][0] = -c/det;
  r.a[1][1] =  a/det;
  return r;
}

inline double wrapAngleRad(double a) {
  while (a > M_PI) a -= 2.0*M_PI;
  while (a < -M_PI) a += 2.0*M_PI;
  return a;
}