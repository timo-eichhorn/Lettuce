#ifndef LETTUCE_STOUT_SMEARING_HPP
#define LETTUCE_STOUT_SMEARING_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../math/su3.hpp"
#include "../math/su3_exp.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
// ...

// To calculate the stout force, we need the following arrays:
// 1 array holding the unsmeared gauge field
// n_smear arrays holding the smeared gauge fields (alternatively one larger array/class)
// 2 arrays holding Sigma and Sigma' (is 1 array enough if we overwrite it in place?)
// 1 array holding Lambda

// void CalculateB(const GaugeField4D<Matrix_SU3>& Gluon)
// {
//     // First calculate r^i_j
//     r1_0 = static_cast<floatT>(2.0) * (u + i<floatT> * (u * u - w * w)) * std::exp(static_cast<floatT>(2.0) * i<floatT> * u)
//          + static_cast<floatT>(2.0) * std::exp(-i<floatT> * u) * (static_cast<floatT>(4.0) * u * (static_cast<floatT>(2.0) - i<floatT> * u) * std::cos(w)
//          + i<floatT> * (static_cast<floatT>(9.0) * u * u + w * w - i<floatT> * u * (static_cast<floatT>(3.0) * u * u + w * w)) * xi_0(w));
//     r1_1 = static_cast<floatT>(2.0) * (static_cast<floatT>(1.0) + static_cast<floatT>(2.0) * i<floatT> * u) * std::exp(static_cast<floatT>(2.0) * i<floatT> * u) + std::exp(-i<floatT> * u) * (-static_cast<floatT>(2.0));
//     r1_2 = static_cast<floatT>(2.0) * i<floatT> * std::exp(static_cast<floatT>(2.0) * i<floatT> * u) + i<floatT> * std::exp(-i<floatT> * u) * (std::cos(w) - static_cast<floatT>(3.0) * (static_cast<floatT>(1.0) - i<floatT> * u) * xi_0(w));
//     r2_0 = -static_cast<floatT>(2.0) * std::exp(static_cast<floatT>(2.0) * i<floatT> * u) + static_cast<floatT>(2.0) * i<floatT> * u * std::exp(-i<floatT> * u) * (std::cos(w) + (static_cast<floatT>(1.0) + static_cast<floatT>(4.0) * i<floatT> * u) * xi_0(w) + static_cast<floatT>(3.0) * u * u * xi_1(w));
//     r2_1 = -i<floatT> * std::exp(-i<floatT> * u) * (std::cos(w) + (static_cast<floatT>(1.0) + static_cast<floatT>(2.0) * i<floatT> * u) * xi_0(w) - static_cast<floatT>(3.0) * u * u * xi_1(w));
//     r2_2 = std::exp(-i<floatT> * u) * (xi_0(w) - static_cast<floatT>(3.0) * i<floatT> * xi_1(w));
//     // Denominator
//     std::complex<floatT> u2 {u * u};
//     std::complex<floatT> w2 {w * w};
//     // static_cast<floatT>(2.0) * u * r
// }

// void CalculateLambda(const GaugeField4D<Matrix_SU3>& Gluon, GaugeField4D<Matrix_SU3>& Lambda)
// {
//     #pragma omp parallel for
//     for (int t = 0; t < Nt; ++t)
//     for (int x = 0; x < Nx; ++x)
//     for (int y = 0; y < Ny; ++y)
//     for (int z = 0; z < Nz; ++z)
//     {
//         Gamma({t, x, y, z, mu})  = (Sigma_prev({t, x, y, z, mu}) * B1 * Gluon({t, x, y, z, mu})).trace() * Q
//                                  + (Sigma_prev({t, x, y, z, mu}) * B2 * Gluon({t, x, y, z, mu})).trace() * Q2
//                                  + f1 * Gluon({t, x, y, z, mu}) * Sigma_prev({t, x, y, z, mu})
//                                  + f2 * (Q * Gluon[t][x][y][z][mu] * Sigma_prev({t, x, y, z, mu}) + Gluon({t, x, y, z, mu}) * Sigma_prev({t, x, y, z, mu}) * Q);
//         Lambda({t, x, y, z, mu}) = SU3::Projection::Algebra(Gamma);
//     }
// }

// void StoutSmearing4DWithForce(const GaugeField4D<Matrix_SU3>& Gluon_unsmeared, GaugeField4D<Matrix_SU3>& Gluon_smeared, const floatT smear_param = 0.12)
// {
//     Matrix_3x3 st;
//     Matrix_3x3 A;
//     Matrix_3x3 B;
//     Matrix_3x3 C;

//     #pragma omp parallel for private(st, A, B, C)
//     for (int t = 0; t < Nt; ++t)
//     for (int x = 0; x < Nx; ++x)
//     for (int y = 0; y < Ny; ++y)
//     for (int z = 0; z < Nz; ++z)
//     {
//         for (int mu = 0; mu < 4; ++mu)
//         {
//             // Stout smearing
//             st.noalias() = WilsonAction::Staple(Gluon_unsmeared, t, x, y, z, mu);
//             A.noalias() = st * Gluon_unsmeared({t, x, y, z, mu}).adjoint();
//             B.noalias() = A - A.adjoint();
//             C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_3x3::Identity();
//             Gluon_smeared({t, x, y, z, mu}) = CH::Exp(-i<floatT> * smear_param * C) * Gluon_unsmeared({t, x, y, z, mu});
//             ProjectionSU3Single(Gluon_smeared({t, x, y, z, mu}));
//             // Stout force
//             for (int nu = 0; nu < 4; ++nu)
//             {
//                 // TODO: Is there a directional symmetry we can use instead of checking this?
//                 // Or perhaps move the directional loops outside?
//                 if (mu != nu)
//                 {
//                     site_coord site_mup     = Move< 1>(current_site, mu);
//                     site_coord site_nup     = Move< 1>(current_site, nu);
//                     site_coord site_nud     = Move<-1>(current_site, nu);
//                     site_coord site_mup_nud = Move<-1>(site_mu     , nu);

//                     force_sum = U(site_mup    , nu)           * U(site_nup    , mu).adjoint() * U(current_site, nu).adjoint()   * L(current_site, nu)
//                               + U(site_mup_nud, nu).adjoint() * U(site_nud    , mu).adjoint() * L(site_nud    , mu)             * U(site_nud, nu)
//                               + U(site_mup_nud, nu).adjoint() * L(site_mup_nud, nu)           * U(site_nud    , mu).adjoint()   * U(site_nud, nu)
//                               - U(site_mup_nud, nu).adjoint() * U(site_nud    , mu).adjoint() * L(site_nud    , nu)             * U(site_nud, nu)
//                               - L(site_mup    , nu)           * U(site_mup    , nu)           * U(site_nup    , mu).adjoint()   * U(current_site, nu).adjoint()
//                               + U(site_mup    , nu)           * U(site_nup    , mu).adjoint() * U(site_nup    , mu)             * U(current_site, nu).adjoint();
//                 }
//             }
//             Sigma({t, x, y, z, mu}) = Sigma_prev({t, x, y, z, mu}) * CH::Exp(-i<floatT> * smear_param * C) + i<floatT> * st.adjoint() * L({t, x, y, z, mu})
//                                   - i<floatT> * smear_param * force_sum;
//         }
//     }
// }

#endif // LETTUCE_STOUT_SMEARING_HPP
