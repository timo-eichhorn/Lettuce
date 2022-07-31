// Test clover charge derivative and action derivative against numerical derivative
// This is the version without any smearing

for (int i = 0; i < 5; ++i)
{
    Iterator::Checkerboard(Heatbath, 1);
    Iterator::Checkerboard(OverrelaxationSubgroup, 4);
}
GaugeField        FieldFwd;
GaugeField        FieldBwd;
GaugeFieldSmeared TestFieldSmeared(n_smear_meta);
FullTensor        Clov;

site_coord        current_site {1, 2, 0, 1};
int               direction    {3};
link_coord        current_link {1, 2, 0, 1, 3};
double            deltah       {0.00001};

double ChargeOld {TopChargeGluonicSymm(Gluon)};
double ActionOld {WilsonAction::Action(Gluon)};

for (int group_direction = 1; group_direction <= 8; ++group_direction)
{
    // Go fowards by deltah in group_direction
    FieldFwd = Gluon;
    FieldFwd(current_site, direction) = SU3::Generators::Exp_lambda(group_direction, deltah) * FieldFwd(current_link);
    double ChargeNewF {TopChargeGluonicSymm(FieldFwd)};
    double ActionNewF {WilsonAction::Action(FieldFwd)};

    // Go backwards by deltah in group_direction
    FieldBwd = Gluon;
    FieldBwd(current_site, direction) = SU3::Generators::Exp_lambda(group_direction, -deltah) * FieldBwd(current_link);
    double ChargeNewB {TopChargeGluonicSymm(FieldBwd)};
    double ActionNewB {WilsonAction::Action(FieldBwd)};

    // Calculate the clover charge derivative
    CalculateClover(Gluon, Clov);
    Matrix_3x3 ClovD {i<floatT> * CloverDerivative(Gluon, Clov, current_site, direction)};

    // Calculate the Wilson action derivative (where the sum over all 8 generators is already taken)
    Matrix_3x3 st {WilsonAction::Staple(Gluon, current_link)};
    Matrix_3x3 tmp {st * Gluon(current_link).adjoint() - Gluon(current_link) * st.adjoint()};
    tmp *= i<floatT> * beta / 12.0 * Matrix_3x3::Identity();

    // Calculate the Wilson action derivative (in only one direction)
    // Need to multiply with factor 0.5 * i since we want the T^a, not the lambda^a (the 0.5 cancels out with the factor 2, see thesis notes)
    Matrix_3x3 tmp2  {i<floatT> * SU3::Generators::lambda(group_direction) * Gluon(current_link) * st.adjoint()};
    double actionD   {-beta/6.0 * std::real(tmp2.trace())};

    std::cout << "Group direction " << group_direction << std::endl;
    std::cout << "Topological charge before: " << ChargeOld << std::endl;
    std::cout << "Topological charge after : " << ChargeNewF << std::endl;
    std::cout << "Difference               : " << (ChargeNewF - ChargeOld) / deltah << std::endl;
    std::cout << "Symmetric difference     : " << 0.5 * (ChargeNewF - ChargeNewB) / deltah << std::endl;
    // std::cout << "Clover derivative: " << ClovD << std::endl;
    std::cout << "Derivative (projected)   : " << (SU3::Generators::lambda(group_direction) * ClovD).trace() << std::endl;
    // std::cout << "Change: " << 1.0/3.0 * (SU3::Generators::lambda(group_direction) * ClovD).trace() << std::endl;
    //-----
    std::cout << "Action before            : " << ActionOld << std::endl;
    std::cout << "Action after             : " << ActionNewF << std::endl;
    std::cout << "Difference               : " << (ActionNewF - ActionOld) / deltah << std::endl;
    std::cout << "Symmetric difference     : " << 0.5 * (ActionNewF - ActionNewB) / deltah << std::endl;
    // std::cout << "Action derivative: " << tmp << std::endl;
    std::cout << "Derivative (projected)   : " << (SU3::Generators::lambda(group_direction) * tmp).trace() << std::endl;
    // std::cout << "Change: " << 1.0/3.0 * (SU3::Generators::lambda(group_direction) * tmp).trace() << std::endl;
    std::cout << "Derivative (partial)     : " << actionD << std::endl;
    std::cout <<"--------------------" << std::endl;
}
std::exit(0);

// The most up to date version which checks smeared quantities
// Using standard parameters (rho_stout = 0.12, 10 smearing levels, deltah = 0.001 or 0.00001) led to relative differences of order 0.001% to 0.0001%

for (int i = 0; i < 5; ++i)
{
    Iterator::Checkerboard(Heatbath, 1);
    Iterator::Checkerboard(OverrelaxationSubgroup, 4);
}

GaugeField        FieldFwd;
GaugeField        FieldBwd;
GaugeFieldSmeared TestFieldSmeared(n_smear_meta + 1);
GaugeField4DSmeared<Nt, Nx, Ny, Nz, SU3::ExpConstants> Exp_consts(n_smear_meta);
GaugeField        TopForceFatLink;
GaugeField        ForceFatLink;
FullTensor        Clov;

site_coord        current_site {1, 2, 0, 1};
int               direction    {3};
link_coord        current_link {1, 2, 0, 1, 3};
double            deltah       {0.001};

// TODO: Difference between manually smearing and using StoutSmearingAll!
// std::cout << TopChargeGluonicSymm(Gluon) << std::endl;
// std::cout << WilsonAction::Action(Gluon) << std::endl;
TestFieldSmeared[0] = Gluon;
StoutSmearingAll(TestFieldSmeared, rho_stout);
// for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
// {
//     StoutSmearing4D(TestFieldSmeared[smear_count], TestFieldSmeared[smear_count + 1], rho_stout);
// }
// std::cout << TopChargeGluonicSymm(TestFieldSmeared[n_smear_meta]) << std::endl;
// std::cout << WilsonAction::Action(TestFieldSmeared[n_smear_meta]) << std::endl;

double ChargeOld        {TopChargeGluonicSymm(Gluon)};
double ChargeSmearedOld {TopChargeGluonicSymm(TestFieldSmeared[n_smear_meta])};
double ActionOld        {WilsonAction::Action(Gluon)};
double ActionSmearedOld {WilsonAction::Action(TestFieldSmeared[n_smear_meta])};

for (int group_direction = 1; group_direction <= 8; ++group_direction)
{
    // Go fowards by deltah in group_direction
    FieldFwd = Gluon;
    FieldFwd(current_site, direction) = SU3::Generators::Exp_lambda(group_direction, deltah) * FieldFwd(current_link);
    TestFieldSmeared[0] = FieldFwd;
    // StoutSmearingAll(TestFieldSmeared, rho_stout);
    for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
    {
        StoutSmearing4D(TestFieldSmeared[smear_count], TestFieldSmeared[smear_count + 1], rho_stout);
    }
    double ChargeNewF        {TopChargeGluonicSymm(FieldFwd)};
    double ChargeSmearedNewF {TopChargeGluonicSymm(TestFieldSmeared[n_smear_meta])};
    double ActionNewF        {WilsonAction::Action(FieldFwd)};
    double ActionSmearedNewF {WilsonAction::Action(TestFieldSmeared[n_smear_meta])};

    // Go backwards by deltah in group_direction
    FieldBwd = Gluon;
    FieldBwd(current_site, direction) = SU3::Generators::Exp_lambda(group_direction, -deltah) * FieldBwd(current_link);
    TestFieldSmeared[0] = FieldBwd;
    // StoutSmearingAll(TestFieldSmeared, rho_stout);
    for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
    {
        StoutSmearing4D(TestFieldSmeared[smear_count], TestFieldSmeared[smear_count + 1], rho_stout);
    }
    double ChargeNewB        {TopChargeGluonicSymm(FieldBwd)};
    double ChargeSmearedNewB {TopChargeGluonicSymm(TestFieldSmeared[n_smear_meta])};
    double ActionNewB        {WilsonAction::Action(FieldBwd)};
    double ActionSmearedNewB {WilsonAction::Action(TestFieldSmeared[n_smear_meta])};

    // Calculate the clover charge derivative (the factor i comes in at the end)
    CalculateClover(Gluon, Clov);
    Matrix_3x3 ClovD {i<floatT> * CloverDerivative(Gluon, Clov, current_site, direction)};

    // Calculate the smeared clover charge derivative (here we don't have a factor i until the very end)
    TestFieldSmeared[0] = Gluon;
    for (int smear_count = 0; smear_count < n_smear_meta; ++smear_count)
    {
        StoutSmearing4DWithConstants(TestFieldSmeared[smear_count], TestFieldSmeared[smear_count + 1], Exp_consts[smear_count], rho_stout);
    }
    CalculateClover(TestFieldSmeared[n_smear_meta], Clov);
    // Original clover derivative on maximally smeared field
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    {
        site_coord current_site {t, x, y, z};
        // TODO: This should be a negative sign, since the force is given by the negative derivative of the potential
        //       There is another minus later on in the momentum update
        TopForceFatLink(current_site, mu) = CloverDerivative(TestFieldSmeared[n_smear_meta], Clov, current_site, mu);
    }
    // Stout force recursion
    for (int smear_count = n_smear_meta; smear_count > 0; --smear_count)
    {
        // TODO: Replace global variable rho_stout with parameter?
        StoutForceRecursion(TestFieldSmeared[smear_count - 1], TestFieldSmeared[smear_count], TopForceFatLink, Exp_consts[smear_count - 1], rho_stout);
        // std::cout << TopForceFatLink({4,2,6,7,1}) << std::endl;
    }


    // Calculate the Wilson action derivative (where the sum over all 8 generators is already taken)
    Matrix_3x3 st {WilsonAction::Staple(Gluon, current_link)};
    Matrix_3x3 tmp {st * Gluon(current_link).adjoint() - Gluon(current_link) * st.adjoint()};
    tmp *= i<floatT> * beta / 12.0 * Matrix_3x3::Identity();

    // Calculate the smeared Wilson action derivative
    #pragma omp parallel for
    for (int t = 0; t < Nt; ++t)
    for (int x = 0; x < Nx; ++x)
    for (int y = 0; y < Ny; ++y)
    for (int z = 0; z < Nz; ++z)
    for (int mu = 0; mu < 4; ++mu)
    {
        site_coord current_site {t, x, y, z};
        Matrix_3x3 st_smeared  {WilsonAction::Staple(TestFieldSmeared[n_smear_meta], current_site, mu)};
        Matrix_3x3 tmp_smeared {st_smeared * TestFieldSmeared[n_smear_meta](current_site, mu).adjoint() - TestFieldSmeared[n_smear_meta](current_site, mu) * st_smeared.adjoint()};
        ForceFatLink(current_site, mu) = beta / 12.0 * tmp_smeared;
    }
    // Stout force recursion
    for (int smear_count = n_smear_meta; smear_count > 0; --smear_count)
    {
        // TODO: Replace global variable rho_stout with parameter?
        StoutForceRecursion(TestFieldSmeared[smear_count - 1], TestFieldSmeared[smear_count], ForceFatLink, Exp_consts[smear_count - 1], rho_stout);
        // std::cout << TopForceFatLink({4,2,6,7,1}) << std::endl;
    }

    // Calculate the Wilson action derivative (in only one direction)
    // Need to multiply with factor 0.5 * i since we want the T^a, not the lambda^a (the 0.5 cancels out with the factor 2, see thesis notes)
    Matrix_3x3 tmp2  {i<floatT> * SU3::Generators::lambda(group_direction) * Gluon(current_link) * st.adjoint()};
    double actionD   {-beta/6.0 * std::real(tmp2.trace())};

    std::cout << "Group direction " << group_direction << std::endl;
    // std::cout << "Topological charge before: " << ChargeOld << std::endl;
    // std::cout << "Topological charge fwd   : " << ChargeNewF << std::endl;
    // std::cout << "Topological charge bwd   : " << ChargeNewB << std::endl;
    // std::cout << "Difference               : " << (ChargeNewF - ChargeOld) / deltah << std::endl;
    // std::cout << "Symmetric difference     : " << 0.5 * (ChargeNewF - ChargeNewB) / deltah << std::endl;
    // // std::cout << "Clover derivative: " << ClovD << std::endl;
    // std::cout << "Derivative (projected)   : " << (SU3::Generators::lambda(group_direction) * ClovD).trace() << std::endl;
    // std::cout << "-----\n";
    //-----
    double clov_smeared_symm_diff  {0.5 * (ChargeSmearedNewF - ChargeSmearedNewB) / deltah};
    double clov_smeared_derivative {std::real((SU3::Generators::lambda(group_direction) * i<floatT> * TopForceFatLink(current_link)).trace())};
    std::cout << "Topological charge before: " << ChargeSmearedOld << std::endl;
    std::cout << "Topological charge fwd   : " << ChargeSmearedNewF << std::endl;
    std::cout << "Topological charge bwd   : " << ChargeSmearedNewB << std::endl;
    std::cout << "Difference               : " << (ChargeSmearedNewF - ChargeSmearedOld) / deltah << std::endl;
    std::cout << "Symmetric difference     : " << clov_smeared_symm_diff << std::endl;
    // std::cout << "Clover derivative: " << ClovD << std::endl;
    std::cout << "Derivative (projected)   : " << clov_smeared_derivative << std::endl;
    std::cout << "Relative difference      : " << 100 * std::abs(clov_smeared_symm_diff - clov_smeared_derivative) / std::max(std::abs(clov_smeared_symm_diff), std::abs(clov_smeared_derivative)) << std::endl;
    std::cout << "-----\n";
    //-----
    // std::cout << "Action before            : " << ActionOld << std::endl;
    // std::cout << "Action fwd               : " << ActionNewF << std::endl;
    // std::cout << "Action bwd               : " << ActionNewB << std::endl;
    // std::cout << "Difference               : " << (ActionNewF - ActionOld) / deltah << std::endl;
    // std::cout << "Symmetric difference     : " << 0.5 * (ActionNewF - ActionNewB) / deltah << std::endl;
    // // std::cout << "Action derivative: " << tmp << std::endl;
    // std::cout << "Derivative (projected)   : " << (SU3::Generators::lambda(group_direction) * tmp).trace() << std::endl;
    // std::cout << "Derivative (partial)     : " << actionD << std::endl;
    // std::cout << "-----\n";
    //-----
    double action_smeared_symm_diff  {0.5 * (ActionSmearedNewF - ActionSmearedNewB) / deltah};
    double action_smeared_derivative {std::real((SU3::Generators::lambda(group_direction) * i<floatT> * ForceFatLink(current_link)).trace())};
    std::cout << "Action before            : " << ActionSmearedOld << std::endl;
    std::cout << "Action fwd               : " << ActionSmearedNewF << std::endl;
    std::cout << "Action bwd               : " << ActionSmearedNewB << std::endl;
    std::cout << "Difference               : " << (ActionSmearedNewF - ActionSmearedOld) / deltah << std::endl;
    std::cout << "Symmetric difference     : " << action_smeared_symm_diff << std::endl;
    // std::cout << "Action derivative: " << tmp << std::endl;
    std::cout << "Derivative (projected)   : " << action_smeared_derivative << std::endl;
    std::cout << "Relative difference      : " << 100 * std::abs(action_smeared_symm_diff - action_smeared_derivative) / std::max(std::abs(action_smeared_symm_diff), std::abs(action_smeared_derivative)) << std::endl;
    // std::cout << "Derivative (partial)     : " << actionD << std::endl;
    std::cout <<"--------------------" << std::endl;
}
std::exit(0);