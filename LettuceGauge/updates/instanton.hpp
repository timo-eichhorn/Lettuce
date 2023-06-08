#ifndef LETTUCE_INSTANTON_HPP
#define LETTUCE_INSTANTON_HPP

// Non-standard library headers
#include "../defines.hpp"
#include "../coords.hpp"
#include "../lattice.hpp"
#include "../math/su3.hpp"
// #include "../actions/gauge/rectangular_action.hpp"
#include "../actions/gauge/wilson_action.hpp"
// #include "../smearing/gradient_flow.hpp"
#include "../smearing/stout_smearing.hpp"
//-----
#include <Eigen/Dense>
//----------------------------------------
// Standard library headers
#include <omp.h>
//----------------------------------------
// Standard C++ headers
#include <complex>
//----------------------------------------
// Standard C headers
#include <cmath>

// The return value of this function may be negative, as it simply calculates the difference between two coordinates while taking periodic boundaries into account
double DifferenceMu(const int lattice_length, const double coord1, const double coord2) noexcept
{
    // Backwards and forwards are measured with respect to the large of the two coordinates coord1 and coord2
    double backwards_distance {std::abs(coord1 - coord2)};
    double forwards_distance  {lattice_length - backwards_distance};
    // Get a signed(!) distance with respect to coord1
    int    sign               {coord1 >= coord2 ? -1 : 1};
    return sign * (backwards_distance < forwards_distance ? -backwards_distance : forwards_distance);
}

// The return value of this function is always >= 0, since it calculates the distance between two coordinates while taking periodic boundaries into account
double DistanceMu(const int lattice_length, const double coord1, const double coord2) noexcept
{
    // Backwards and forwards are measured with respect to the large of the two coordinates coord1 and coord2
    double backwards_distance {std::abs(coord1 - coord2)};
    double forwards_distance  {lattice_length - backwards_distance};
    return std::fmin(backwards_distance, forwards_distance);
}

double SquaredDistanceToCenter(const site_coord& lattice_shape, const site_coord& center, const site_coord& site) noexcept
{
    // Since we count from 0, coord_max is actually the maximum possible value + 1
    // Also, since we want to avoid gauge singularities, we shift all components of center by 0.5 into the positive direction
    double squared_distance {0.0};
    for (int mu = 0; mu < 4; ++mu)
    {
        double distance_mu {DistanceMu(lattice_shape[mu], center[mu] + 0.5, site[mu])};
        squared_distance += distance_mu * distance_mu;
    }
    return squared_distance;
}

// This creates an approximate (charge 1) BPST instanton on the lattice with scale parameter r
// Note that no self-dual instanton configurations with abs(Q) = 1 exist on the torus (cf. hep-lat/0112034), but for
// the purpose of an attempted instanton update we will try to proceed with an approximate abs(Q) = 1 'instanton'
void CreateBPSTInstanton(GaugeField& Gluon, GaugeField& Gluon1, const bool positive_Q, const site_coord& center, const int r) noexcept
{
    Matrix_SU3 sig1, sig2, sig3;
    sig1 << 0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0;
    sig2 << 0.0      , -i<floatT>, 0.0,
            i<floatT>,  0.0      , 0.0,
            0.0      ,  0.0      , 1.0;
    sig3 << 1.0,  0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0,  0.0, 1.0;
    // sig1 << 1.0, 0.0, 0.0,
    //         0.0, 0.0, 1.0,
    //         0.0, 1.0, 0.0;
    // sig2 << 1.0, 0.0      ,  0.0,
    //         0.0, 0.0      , -i<floatT>,
    //         0.0, i<floatT>,  0.0;
    // sig3 << 1.0, 0.0,  0.0,
    //         0.0, 1.0,  0.0,
    //         0.0, 0.0, -1.0;
    floatT sign;
    if (positive_Q)
    {
        sign = -1.0;
    }
    else
    {
        sign = 1.0;
    }
    // TODO: Overload +/- operators on site_coord to calculate distances?
    // To avoid gauge singularities on the lattice, we actually do not place the instanton around the site_coord center, but rather shift all coordinates by 0.5 into the positive direction
    // This way, the gauge singularity at the center of the instanton never actually coincides with a lattice point
    // Also, due to periodic boundaries
    site_coord lattice_shape {Gluon.Shape()};
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    {
        site_coord current_site        {t, x, y, z};
        double     distance_squared    {SquaredDistanceToCenter(lattice_shape, center, current_site)};
        double     difference_mu       {DifferenceMu(lattice_shape[mu], center[mu] + 0.5, current_site[mu])};
        double     distance_mu_squared {difference_mu * difference_mu};
        double     denom               {std::sqrt(distance_squared - distance_mu_squared)};
        double     denom_r             {std::sqrt(distance_squared - distance_mu_squared + r * r)};
        double     lambda              {-std::atan((difference_mu + 1.0) / denom) + std::atan(difference_mu / denom) + denom / denom_r * (std::atan((difference_mu + 1.0) / denom_r) - std::atan(difference_mu / denom_r))};
        // TODO: Allow to choose between different embeddings
        //       For now only embed in 01 entries of SU(3) matrix
        // TODO: Use SU(2) matrices instead of SU(3) for as long as possible?
        Matrix_SU3 tmp {Matrix_SU3::Zero()};
        switch (mu)
        {
            // Manually worked out the different contributions from the 't Hooft symbol
            case 0:
            {
                // a = 1, mu = 0, nu = 1 => +1
                tmp += sign * sig1 * DifferenceMu(lattice_shape[1], current_site[1], center[1] + 0.5);
                // a = 2, mu = 0, nu = 2 => +1
                tmp += sign * sig2 * DifferenceMu(lattice_shape[2], current_site[2], center[2] + 0.5);
                // a = 3, mu = 0, nu = 3 => +1
                tmp += sign * sig3 * DifferenceMu(lattice_shape[3], current_site[3], center[3] + 0.5);
            }
            break;
            case 1:
            {
                // a = 1, mu = 1, nu = 0 => -1
                tmp -= sign * sig1 * DifferenceMu(lattice_shape[0], current_site[0], center[0] + 0.5);
                // a = 2, mu = 1, nu = 3 => -1
                tmp -=        sig2 * DifferenceMu(lattice_shape[3], current_site[3], center[3] + 0.5);
                // a = 3, mu = 1, nu = 2 => +1
                tmp +=        sig3 * DifferenceMu(lattice_shape[2], current_site[2], center[2] + 0.5);
            }
            break;
            case 2:
            {
                // a = 2, mu = 2, nu = 0 => -1
                tmp -= sign * sig2 * DifferenceMu(lattice_shape[0], current_site[0], center[0] + 0.5);
                // a = 1, mu = 2, nu = 3 => +1
                tmp +=        sig1 * DifferenceMu(lattice_shape[3], current_site[3], center[3] + 0.5);
                // a = 3, mu = 2, nu = 1 => -1
                tmp -=        sig3 * DifferenceMu(lattice_shape[1], current_site[1], center[1] + 0.5);
            }
            break;
            case 3:
            {
                // a = 3, mu = 3, nu = 0 => -1
                tmp -= sign * sig3 * DifferenceMu(lattice_shape[0], current_site[0], center[0] + 0.5);
                // a = 1, mu = 3, nu = 2 => -1
                tmp -=        sig1 * DifferenceMu(lattice_shape[2], current_site[2], center[2] + 0.5);
                // a = 2, mu = 3, nu = 1 => +1
                tmp +=        sig2 * DifferenceMu(lattice_shape[1], current_site[1], center[1] + 0.5);
            }
            break;
        }
        Gluon(current_site, mu) = std::cos(lambda) * Matrix_SU3::Identity() + i<floatT> * std::sin(lambda) / denom * tmp;
        SU3::Projection::GramSchmidt(Gluon(current_site, mu));
    }
    // std::cout << "All elements in group: " << SU3::Tests::TestSU3All(Gluon) << std::endl;
    //-----
    // For now only smear uniformly
    // For even smearing steps the final field is stored in the first argument, so for now we'll hardcode an even amount of smearing steps here (my talk is tomorrow after all)
    StoutSmearingN(Gluon, Gluon1, 150);
    //-----
    // Smearing with spatial dependence (don't smear close to the instanton center, smear more further outwards)
    // Matrix_3x3 Sigma;
    // Matrix_3x3 A;
    // Matrix_3x3 B;
    // Matrix_3x3 C;
    // for(int smear_count = 0; smear_count < 10; ++smear_count)
    // {
    //     #pragma omp parallel for private(Sigma, A, B, C)
    //     for (int t = 0; t < Nt; ++t)
    //     for (int x = 0; x < Nx; ++x)
    //     for (int y = 0; y < Ny; ++y)
    //     for (int z = 0; z < Nz; ++z)
    //     for (int mu = 0; mu < 4; ++mu)
    //     {
    //         link_coord current_link {t, x, y, z, mu};
    //         // TODO: This (hopefully) works as long as we place the instanton around the center of the lattice, but it is probably wrong otherwise since it doesn't respect the periodic boundaries
    //         double     distance     {std::sqrt(std::pow(current_link.t - (center.t + 0.5), 2) + std::pow(current_link.x - (center.x + 0.5), 2) + std::pow(current_link.y - (center.y + 0.5), 2) + std::pow(current_link.z - (center.z + 0.5), 2))};
    //         if (distance >= Nt / 4 and distance <= Nt / 2)
    //         {
    //             double rho_prime {0.5 * rho_stout * (1.0 + std::sin(4.0 * pi<floatT> / static_cast<floatT>(Nt) * (distance - 3.0/8.0 * Nt)))};
    //             Sigma.noalias() = WilsonAction::Staple(Gluon, current_link);
    //             A.noalias() = Sigma * Gluon(current_link).adjoint();
    //             B.noalias() = A - A.adjoint();
    //             C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_SU3::Identity();
    //             // Gluon(current_link) = SU3::exp(-i<floatT> * rho_prime * C) * Gluon(current_link);
    //             Gluon(current_link) = (rho_prime * C).exp() * Gluon(current_link);
    //             SU3::Projection::GramSchmidt(Gluon(current_link));
    //         }
    //         if (distance > Nt / 2)
    //         {
    //             Sigma.noalias() = WilsonAction::Staple(Gluon, current_link);
    //             A.noalias() = Sigma * Gluon(current_link).adjoint();
    //             B.noalias() = A - A.adjoint();
    //             C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_SU3::Identity();
    //             // Gluon(current_link) = SU3::exp(-i<floatT> * rho_prime * C) * Gluon(current_link);
    //             Gluon(current_link) = (rho_stout * C).exp() * Gluon(current_link);
    //             SU3::Projection::GramSchmidt(Gluon(current_link));
    //         }
    //     }
    // }
    // std::cout << Gluon(3, 3, 3, 3, 1) << std::endl;
    // std::cout << Gluon(3, 3, 3, 3, 1) * Gluon(3, 3, 3, 3, 1).adjoint() << std::endl;
    // std::cout << Gluon(3, 3, 3, 3, 1).determinant() << "\n" << std::endl;
    // std::cout << SU3::Tests::TestSU3All(Gluon) << std::endl;
}

void CreateBPSTInstantonOld(GaugeField& Gluon, GaugeField& Gluon1, const bool positive_Q, const site_coord& center, const int r) noexcept
{
    Matrix_SU3 sig1, sig2, sig3;
    sig1 << 0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0;
    sig2 << 0.0      , -i<floatT>, 0.0,
            i<floatT>,  0.0      , 0.0,
            0.0      ,  0.0      , 1.0;
    sig3 << 1.0,  0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0,  0.0, 1.0;
    floatT sign;
    if (positive_Q)
    {
        sign = -1.0;
    }
    else
    {
        sign = 1.0;
    }
    // TODO: Overload +/- operators on site_coord to calculate distances?
    // To avoid gauge singularities on the lattice, we actually do not place the instanton around center, but rather shift all coordinates by 0.5 into the positive direction
    // This way, the gauge singularity at the center of the instanton never actually coincides with a lattice point
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    {
        site_coord current_site {t, x, y, z};
        double     distance2    {std::pow(current_site.t - (center.t + 0.5), 2) + std::pow(current_site.x - (center.x + 0.5), 2) + std::pow(current_site.y - (center.y + 0.5), 2) + std::pow(current_site.z - (center.z + 0.5), 2)};
        double     distance_mu  {current_site[mu] - (center[mu] + 0.5)};
        double     distance_mu2 {distance_mu * distance_mu};
        double     denom        {std::sqrt(distance2 - distance_mu2)};
        double     denom_r      {std::sqrt(distance2 - distance_mu2 + r * r)};
        double     lambda       {-std::atan((distance_mu + 1.0) / denom) + std::atan(distance_mu / denom) + denom / denom_r * (std::atan((distance_mu + 1.0) / denom_r) - std::atan(distance_mu / denom_r))};
        // TODO: Allow to choose between different embeddings
        //       For now only embed in 01 entries of SU(3) matrix
        // TODO: Use SU(2) matrices instead of SU(3) for as long as possible?
        Matrix_SU3 tmp {Matrix_SU3::Zero()};
        switch (mu)
        {
            case 0:
            {
                // a = 1, mu = 0, nu = 1 => +1
                tmp += sign * sig1 * (current_site[1] - center[1]);
                // a = 2, mu = 0, nu = 2 => +1
                tmp += sign * sig2 * (current_site[2] - center[2]);
                // a = 3, mu = 0, nu = 3 => +1
                tmp += sign * sig3 * (current_site[3] - center[3]);
            }
            break;
            case 1:
            {
                // a = 1, mu = 1, nu = 0 => -1
                tmp -= sign * sig1 * (current_site[0] - center[0]);
                // a = 2, mu = 1, nu = 3 => -1
                tmp -=        sig2 * (current_site[3] - center[3]);
                // a = 3, mu = 1, nu = 2 => +1
                tmp +=        sig3 * (current_site[2] - center[2]);
            }
            break;
            case 2:
            {
                // a = 2, mu = 2, nu = 0 => -1
                tmp -= sign * sig2 * (current_site[0] - center[0]);
                // a = 1, mu = 2, nu = 3 => +1
                tmp +=        sig1 * (current_site[3] - center[3]);
                // a = 3, mu = 2, nu = 1 => -1
                tmp -=        sig3 * (current_site[1] - center[1]);
            }
            break;
            case 3:
            {
                // a = 3, mu = 3, nu = 0 => -1
                tmp -= sign * sig3 * (current_site[0] - center[0]);
                // a = 1, mu = 3, nu = 2 => -1
                tmp -=        sig1 * (current_site[2] - center[2]);
                // a = 2, mu = 3, nu = 1 => +1
                tmp +=        sig2 * (current_site[1] - center[1]);
            }
            break;
        }
        Gluon(current_site, mu) = std::cos(lambda) * Matrix_SU3::Identity() + i<floatT> * std::sin(lambda) / denom * tmp;
        SU3::Projection::GramSchmidt(Gluon(current_site, mu));
    }
    // std::cout << SU3::Tests::TestSU3All(Gluon) << std::endl;
    //-----
    // For now only smear uniformly
    // For even smearing steps the final field is stored in the first argument, so for now we'll hardcode an even amount of smearing steps here (my talk is tomorrow after all)
    StoutSmearingN(Gluon, Gluon1, 150);
    //-----
    // Smearing with spatial dependence (don't smear close to the instanton center, smear more further outwards)
    // Matrix_3x3 Sigma;
    // Matrix_3x3 A;
    // Matrix_3x3 B;
    // Matrix_3x3 C;
    // for(int smear_count = 0; smear_count < 10; ++smear_count)
    // {
    //     #pragma omp parallel for private(Sigma, A, B, C)
    //     for (int t = 0; t < Nt; ++t)
    //     for (int x = 0; x < Nx; ++x)
    //     for (int y = 0; y < Ny; ++y)
    //     for (int z = 0; z < Nz; ++z)
    //     for (int mu = 0; mu < 4; ++mu)
    //     {
    //         link_coord current_link {t, x, y, z, mu};
    //         // TODO: This (hopefully) works as long as we place the instanton around the center of the lattice, but it is probably wrong otherwise since it doesn't respect the periodic boundaries
    //         double     distance     {std::sqrt(std::pow(current_link.t - (center.t + 0.5), 2) + std::pow(current_link.x - (center.x + 0.5), 2) + std::pow(current_link.y - (center.y + 0.5), 2) + std::pow(current_link.z - (center.z + 0.5), 2))};
    //         if (distance >= Nt / 4 and distance <= Nt / 2)
    //         {
    //             double rho_prime {0.5 * rho_stout * (1.0 + std::sin(4.0 * pi<floatT> / static_cast<floatT>(Nt) * (distance - 3.0/8.0 * Nt)))};
    //             Sigma.noalias() = WilsonAction::Staple(Gluon, current_link);
    //             A.noalias() = Sigma * Gluon(current_link).adjoint();
    //             B.noalias() = A - A.adjoint();
    //             C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_SU3::Identity();
    //             // Gluon(current_link) = SU3::exp(-i<floatT> * rho_prime * C) * Gluon(current_link);
    //             Gluon(current_link) = (rho_prime * C).exp() * Gluon(current_link);
    //             SU3::Projection::GramSchmidt(Gluon(current_link));
    //         }
    //         if (distance > Nt / 2)
    //         {
    //             Sigma.noalias() = WilsonAction::Staple(Gluon, current_link);
    //             A.noalias() = Sigma * Gluon(current_link).adjoint();
    //             B.noalias() = A - A.adjoint();
    //             C.noalias() = static_cast<floatT>(0.5) * B - static_cast<floatT>(1.0/6.0) * B.trace() * Matrix_SU3::Identity();
    //             // Gluon(current_link) = SU3::exp(-i<floatT> * rho_prime * C) * Gluon(current_link);
    //             Gluon(current_link) = (rho_stout * C).exp() * Gluon(current_link);
    //             SU3::Projection::GramSchmidt(Gluon(current_link));
    //         }
    //     }
    // }
    // std::cout << Gluon(3, 3, 3, 3, 1) << std::endl;
    // std::cout << Gluon(3, 3, 3, 3, 1) * Gluon(3, 3, 3, 3, 1).adjoint() << std::endl;
    // std::cout << Gluon(3, 3, 3, 3, 1).determinant() << "\n" << std::endl;
    // std::cout << SU3::Tests::TestSU3All(Gluon) << std::endl;
}

void MultiplyConfigurations(GaugeField& Gluon1, const GaugeField& Gluon2) noexcept
{
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    {
        link_coord current_link {t, x, y, z, mu};
        Gluon1(current_link) = Gluon1(current_link) * Gluon2(current_link);
    }
}

bool BPSTInstantonUpdate(GaugeField& Gluon, GaugeField& Gluon_copy, const int Q, const site_coord& center, const int r, uint_fast64_t& acceptance_count_instanton, const bool metropolis_test, std::uniform_real_distribution<floatT>& distribution_prob, const bool create_instantons = false) noexcept
{
    static GaugeField PositiveInstanton;
    static GaugeField NegativeInstanton;
    static GaugeField Gluonsmeared_tmp;
    if (create_instantons)
    {
        CreateBPSTInstanton(PositiveInstanton, Gluonsmeared_tmp, true, center, r);
        CreateBPSTInstanton(NegativeInstanton, Gluonsmeared_tmp, false, center, r);
    }
    // Actual update
    Gluon_copy = Gluon;
    if (Q == 1)
    {
        MultiplyConfigurations(Gluon_copy, PositiveInstanton);
    }
    if (Q == -1)
    {
        MultiplyConfigurations(Gluon_copy, NegativeInstanton);
    }
    double S_old {WilsonAction::Action(Gluon)};
    double S_new {WilsonAction::Action(Gluon_copy)};
    double p     {std::exp(-S_new + S_old)};
    #if defined(_OPENMP)
    double q     {distribution_prob(prng_vector[omp_get_thread_num()])};
    #else
    double q     {distribution_prob(generator_rand)};
    #endif
    // TODO: Probably shouldnt use a global variable for DeltaSInstanton?
    DeltaSInstanton = S_new - S_old;
    if (metropolis_test)
    {
        if (q <= p)
        {
            Gluon = Gluon_copy;
            acceptance_count_instanton += 1;
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        Gluon = Gluon_copy;
        // datalog << "DeltaS: " << DeltaS << std::endl;
        return true;
    }
}

// Old stuff that is not really correct

//-----
// Creates maximally dislocated instanton configuration with charge Q_i

// void InstantonStart(Gl_Lattice& Gluon, const int Q, const int mu, const int nu, const int sigma, const int tau)
void InstantonStart(GaugeField& Gluon, const int Q)
{
    // First set everything to unity
    Gluon.SetToIdentity();
    // Commuting su(2) matrices embedded into su(3)
    // Generally, any traceless, hermitian matrix works here
    // The condition that the matrices commute enables us to construct charge +/- 1 instantons
    Matrix_3x3 sig;
    sig << 1,  0, 0,
           0, -1, 0,
           0,  0, 0;
    Matrix_3x3 tau;
    tau << 1, 0,  0,
           0, 0,  0,
           0, 0, -1;
    // Unit matrices in sigma and tau subspace
    Matrix_3x3 id_sig {sig * sig};
    Matrix_3x3 id_tau {tau * tau};
    // Orthogonal complement
    Matrix_3x3 comp_sig {Matrix_3x3::Identity() - id_sig};
    Matrix_3x3 comp_tau {Matrix_3x3::Identity() - id_tau};
    // Consider a two-dimensional slice in the t-x plane
    floatT Field_t {static_cast<floatT>(2.0) * pi<floatT> * std::abs(Q) / (Nt * Nx)};
    floatT Field_x {static_cast<floatT>(2.0) * pi<floatT> * std::abs(Q) / Nx};
    // Assign link values in t-direction
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        Gluon({t, x, y, z, 1}) = comp_sig + std::cos(Field_t * t) * id_sig + i<floatT> * std::sin(Field_t * t) * sig;
    }
    // Assign link values on last time-slice in x-direction
    #pragma omp parallel for
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        Gluon({Nt - 1, x, y, z, 0}) = comp_sig + std::cos(Field_x * x) * id_sig - i<floatT> * std::sin(Field_x * x) * sig;
    }
    //-----
    // Consider a two-dimensional slice in the y-z plane
    floatT Field_y, Field_z;
    if (Q == 0)
    {
        Field_y = static_cast<floatT>(0.0);
        Field_z = static_cast<floatT>(0.0);
    }
    else
    {
        Field_y = static_cast<floatT>(2.0) * pi<floatT> * Q / (std::abs(Q) * Ny * Nz);
        Field_z = static_cast<floatT>(2.0) * pi<floatT> * Q / (std::abs(Q) * Nz);
    }
    // Assign link values in y-direction
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        Gluon({t, x, y, z, 3}) = comp_tau + std::cos(Field_y * y) * id_tau + i<floatT> * std::sin(Field_y * y) * tau;
    }
    // Assign link values on last y-slice in z-direction
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int z = 0; z < Nz; ++z)
    {
        Gluon({t, x, Ny - 1, z, 2}) = comp_tau + std::cos(Field_z * z) * id_tau - i<floatT> * std::sin(Field_z * z) * tau;
    }
}

//-----
// TODO: Still WIP
// Creates a local instanton configuration with charge ?

void LocalInstantonStart(GaugeField& Gluon)
{
    // First set everything to unity
    Gluon.SetToIdentity();
    // Generators of SU(2)/Pauli matrices embedded into SU(3) (up to negative determinant)
    // Generally, any traceless, hermitian matrix works here
    // The condition that the matrices commute enables us to construct charge +/- 1 instantons
    // TODO: This is incorrect, the factors only apply to the stuff inside SU(2), the last entry is always 1!
    // TODO: Directly initialize?
    Matrix_3x3 sig1;
    sig1 << 0, 1, 0,
            1, 0, 0,
            0, 0, 1;
    Matrix_3x3 sig2;
    sig2 << 0        , -i<floatT>, 0,
            i<floatT>,  0        , 0,
            0        ,  0        , 1;
    Matrix_3x3 sig3;
    sig3 << 1,  0, 0,
            0, -1, 0,
            0,  0, 1;
    // Only links in the elementary hypercube between (0,0,0,0) and (1,1,1,1) take on non-trivial values
    // TODO: See above!
    for (int t = 0; t < 2; ++t)
    for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
    for (int z = 0; z < 2; ++z)
    {
        int coord_sum {t + x + y + z};
        // We only consider ther 32 links INSIDE the hypercube between (0,0,0,0) and (1,1,1,1)
        // Do not go into mu direction if x_mu != 0
        if (x == 0)
        {
            Gluon({t, x, y, z, 1}) = i<floatT> * std::pow(-1, coord_sum) * sig1;
            Gluon({t, x, y, z, 1})(2, 2) = 1.0;
        }
        if (y == 0)
        {
            Gluon({t, x, y, z, 2}) = i<floatT> * std::pow(-1, coord_sum) * sig2;
            Gluon({t, x, y, z, 2})(2, 2) = 1.0;
        }
        if (z == 0)
        {
            Gluon({t, x, y, z, 3}) = i<floatT> * std::pow(-1, coord_sum) * sig3;
            Gluon({t, x, y, z, 3})(2, 2) = 1.0;
        }
    }
}

//-----
// Instanton update?
// TODO: Rearrange for-loops so we don't unnecessarily loop over the lattice

void MultiplyInstanton(GaugeField& Gluon, const int Q)
{
    double action_old {WilsonAction::Action(Gluon)};
    // Embedded commuting SU(2) matrices
    // Generally, any traceless, hermitian matrix works here
    // The condition that the matrices commute enables us to construct charge +/- 1 instantons
    Matrix_3x3 sig;
    sig << 1,  0, 0,
           0, -1, 0,
           0,  0, 0;
    Matrix_3x3 tau;
    tau << 1, 0,  0,
           0, 0,  0,
           0, 0, -1;
    // Unit matrices in sigma and tau subspace
    Matrix_3x3 id_sig {sig * sig};
    Matrix_3x3 id_tau {tau * tau};
    // Orthogonal complement
    Matrix_3x3 comp_sig {Matrix_3x3::Identity() - id_sig};
    Matrix_3x3 comp_tau {Matrix_3x3::Identity() - id_tau};
    // Consider a two-dimensional slice in the t-x plane
    floatT Field_t {static_cast<floatT>(2.0) * pi<floatT> * std::abs(Q) / (Nt * Nx)};
    floatT Field_x {static_cast<floatT>(2.0) * pi<floatT> * std::abs(Q) / Nx};
    // Assign link values in t-direction
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        Gluon({t, x, y, z, 1}) *= comp_sig + std::cos(Field_t * t) * id_sig + i<floatT> * std::sin(Field_t * t) * sig;
    }
    // Assign link values on last time-slice in x-direction
    #pragma omp parallel for
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        Gluon({Nt - 1, x, y, z, 0}) *= comp_sig + std::cos(Field_x * x) * id_sig - i<floatT> * std::sin(Field_x * x) * sig;
    }
    //-----
    // Consider a two-dimensional slice in the y-z plane
    floatT Field_y, Field_z;
    if (Q == 0)
    {
        Field_y = static_cast<floatT>(0.0);
        Field_z = static_cast<floatT>(0.0);
    }
    else
    {
        Field_y = static_cast<floatT>(2.0) * pi<floatT> * Q / (std::abs(Q) * Ny * Nz);
        Field_z = static_cast<floatT>(2.0) * pi<floatT> * Q / (std::abs(Q) * Nz);
    }
    // Assign link values in y-direction
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    {
        Gluon({t, x, y, z, 3}) *= comp_tau + std::cos(Field_y * y) * id_tau + i<floatT> * std::sin(Field_y * y) * tau;
    }
    // Assign link values on last y-slice in z-direction
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int z = 0; z < Nz; ++z)
    {
        Gluon({t, x, Ny - 1, z, 2}) *= comp_tau + std::cos(Field_z * z) * id_tau - i<floatT> * std::sin(Field_z * z) * tau;
    }
    double action_new {WilsonAction::Action(Gluon)};
    double action_diff {action_new - action_old};
    std::cout << "Action (old): " << action_old << "\n";
    std::cout << "Action (new): " << action_new << "\n";
    std::cout << "Action difference: " << action_diff << "\n";
    std::cout << "Acceptance probability: " << std::exp(-action_diff) << "\n" << std::endl;
}

void MultiplyLocalInstanton(GaugeField& Gluon)
{
    double action_old {WilsonAction::Action(Gluon)};
    // Generators of SU(2)/Pauli matrices embedded into SU(3) (up to negative determinant)
    // Generally, any traceless, hermitian matrix works here
    // The condition that the matrices commute enables us to construct charge +/- 1 instantons
    // TODO: This is incorrect, the factors only apply to the stuff inside SU(2), the last entry is always 1!
    // TODO: Directly initialize?
    Matrix_3x3 sig1;
    sig1 << 0, 1, 0,
            1, 0, 0,
            0, 0, 1;
    Matrix_3x3 sig2;
    sig2 << 0        , -i<floatT>, 0,
            i<floatT>,  0        , 0,
            0        ,  0        , 1;
    Matrix_3x3 sig3;
    sig3 << 1,  0, 0,
            0, -1, 0,
            0,  0, 1;
    // Only links in the elementary hypercube between (0,0,0,0) and (1,1,1,1) take on non-trivial values
    // TODO: See above!
    Matrix_SU3 tmp;
    for (int t = 0; t < 2; ++t)
    for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
    for (int z = 0; z < 2; ++z)
    {
        int coord_sum {t + x + y + z};
        // We only consider ther 32 links INSIDE the hypercube between (0,0,0,0) and (1,1,1,1)
        // Do not go into mu direction if x_mu != 0
        if (x == 0)
        {
            tmp = i<floatT> * std::pow(-1, coord_sum) * sig1;
            tmp(2, 2) = 1.0;
            Gluon({t, x, y, z, 1}) = Gluon({t, x, y, z, 1}) * tmp;
        }
        if (y == 0)
        {
            tmp = i<floatT> * std::pow(-1, coord_sum) * sig2;
            tmp(2, 2) = 1.0;
            Gluon({t, x, y, z, 2}) = Gluon({t, x, y, z, 2}) * tmp;
        }
        if (z == 0)
        {
            tmp = i<floatT> * std::pow(-1, coord_sum) * sig3;
            tmp(2, 2) = 1.0;
            Gluon({t, x, y, z, 3}) = Gluon({t, x, y, z, 3}) * tmp;
        }
    }
    double action_new {WilsonAction::Action(Gluon)};
    double action_diff {action_new - action_old};
    std::cout << "Action (old): " << action_old << "\n";
    std::cout << "Action (new): " << action_new << "\n";
    std::cout << "Action difference: " << action_diff << "\n";
    std::cout << "Acceptance probability: " << std::exp(-action_diff) << "\n" << std::endl;
}

#endif // LETTUCE_INSTANTON_HPP
