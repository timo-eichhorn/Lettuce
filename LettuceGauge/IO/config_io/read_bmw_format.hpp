#ifndef LETTUCE_READ_BMW_FORMAT_HPP
#define LETTUCE_READ_BMW_FORMAT_HPP

// Non-standard library headers
#include "../../math/su3.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <complex>
#include <string>
// #include <fstream>
//----------------------------------------
// Standard C headers
#include <cstdio>

// TODO: Lot's of cleanup still left to do here...
//       Also benchmark how different buffer sizes (12 vs N_mu * 12) affect the performance

// bool CheckFormatBMW(const std::string& filename)
// {
//     // TODO: Implement
// }

template<typename T>
void SwapEndianess(T* in) noexcept
{
    char* const p = reinterpret_cast<char*>(in);
    for (std::size_t i = 0; i < sizeof(T) / 2; ++i)
    {
        char tmp = p[i];
        p[i] = p[sizeof(T) - i - 1];
        p[sizeof(T) - i - 1] = tmp;
    }
}

Matrix_SU3 ReconstructMatBMW(const std::array<double, 12>& buffer) noexcept
{
    // BMW format stores only the first two rows
    Matrix_SU3 tmp;
    tmp << buffer[0] + i<floatT> * buffer[1], buffer[2] + i<floatT> * buffer[3], buffer[4] + i<floatT> * buffer[5],
           buffer[6] + i<floatT> * buffer[7], buffer[8] + i<floatT> * buffer[9], buffer[10] + i<floatT> * buffer[11],
           0.0, 0.0, 0.0;
    SU3::Projection::RestoreLastRow(tmp);
    return tmp;
}

bool ReadConfigBMW(GaugeField& U, const std::string& filename)
{
    // TODO: Reorder for better alignment?
    // std::ifstream         config_stream;
    const std::size_t     header_block_size {4096};
    const std::size_t     object_size {1};
    char                  header_block[header_block_size];
    site_coord            lattice_lengths {0, 0, 0, 0};
    char                  checksum_read[32];
    char                  checksum_new[32];
    int                   header_characters_read {0};

    // Attempt to open file and check stream status
    // config_stream.open(filename, std::fstream::in);
    // if (config_stream.fail())
    // {
    //     std::cout << Lettuce::Color::BoldRed << "Error while opening file " << filename << "!" << Lettuce::Color::Reset << std::endl;
    //     // return false;
    //     return;
    // }

    std::cout << Lettuce::Color::BoldBlue << "Attempting to read configuration in BMW format from " << filename << ":" << Lettuce::Color::Reset << std::endl;

    const char* filename_cstr   = filename.c_str();
    std::FILE*  file            = std::fopen(filename_cstr, "r");
    std::string indent_whitespace {"    "};
    auto        start_read_header {std::chrono::high_resolution_clock::now()};

    // Read header block and check if file matches expected format
    // config_stream.read(reinterpret_cast<char*>(), header_block_size)
    std::size_t successful_reads   = std::fread(header_block, object_size, header_block_size, file);
    // BMW format stores the lattice lenghts in the following order: (x, y, z, t)
    std::size_t successful_assigns = std::sscanf(header_block, "#BMW %d %d %d %d %31s %n", &lattice_lengths.x, &lattice_lengths.y, &lattice_lengths.z, &lattice_lengths.t, checksum_read, &header_characters_read);
    if (successful_reads != header_block_size or successful_assigns < 5)
    {
        std::cout << Lettuce::Color::BoldRed << indent_whitespace << "Header block does not match expected format!" << Lettuce::Color::Reset << std::endl;
        return false;
    }

    // Check if lattice size in file matches the current lattice size
    bool lengths_match {true};
    for(int mu = 0; mu < 4; ++mu)
    {
        lengths_match &= (U.Length(mu) == lattice_lengths[mu]);
    }
    if (not lengths_match)
    {
        std::cout << Lettuce::Color::BoldRed << indent_whitespace << "Lattice sizes do not match!\n";
        std::cout                            << indent_whitespace << "Current sizes: " << Nt << ", " << Nx << ", " << Ny << ", " << Nz << "\n";
        std::cout                            << indent_whitespace << lattice_lengths << Lettuce::Color::Reset << std::endl;
        return false;
    }
    std::cout << Lettuce::Color::BoldBlue << indent_whitespace << "Reading lattice from " << filename << "..." << Lettuce::Color::Reset << std::endl;
    auto end_read_header {std::chrono::high_resolution_clock::now()};

    auto start_read_config {std::chrono::high_resolution_clock::now()};
    std::array<double, 12> buffer;
    // BMW format link order corresponds to array[Nt][Nz][Ny][Nx], i.e., t is the slowest and x the fastest index
    for (int t = 0; t < Nt; ++t)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y)
    for (int x = 0; x < Nx; ++x)
    // This version is not really faster... (remember to change buffer size to 48 above!)
    // {
    //     // Read all four mu-directions at once
    //     std::fread(&buffer, sizeof(double), 48, file);
    //     for (int i = 0; i < 48; ++i)
    //     {
    //         SwapEndianess(buffer + i);
    //     }
    //     U({t, x, y, z, 1}) = ReconstructMatBMW(buffer);
    //     U({t, x, y, z, 2}) = ReconstructMatBMW(buffer + 12);
    //     U({t, x, y, z, 3}) = ReconstructMatBMW(buffer + 24);
    //     U({t, x, y, z, 0}) = ReconstructMatBMW(buffer + 36);
    // }
    for (auto mu : {1, 2, 3, 0})
    {
        std::fread(buffer.data(), sizeof(double), 12, file);
        for (int i = 0; i < 12; ++i)
        {
            SwapEndianess(buffer.data() + i);
        }
        U({t, x, y, z, mu}) = ReconstructMatBMW(buffer);
    }
    auto end_read_config {std::chrono::system_clock::now()};
    std::chrono::duration<double> read_time_header {end_read_header - start_read_header};
    std::chrono::duration<double> read_time_config {end_read_config - start_read_config};
    std::cout << indent_whitespace << "Time for reading header: " << read_time_header.count() << "\n";
    std::cout << indent_whitespace << "Time for reading config: " << read_time_config.count() << std::endl;

    bool InGroup {SU3::Tests::TestSU3All(U)};
    if (InGroup)
    {
        std::cout << Lettuce::Color::BoldGreen << indent_whitespace << "All elements are in SU(3)!" << Lettuce::Color::Reset << std::endl;
        return true;
    }
    else
    {
        std::cout << Lettuce::Color::BoldRed << indent_whitespace << "Not all elements are in SU(3)!" << Lettuce::Color::Reset << std::endl;
        return false;
    }
}

// bool SaveConfigBMW(GaugeField& U, const std::string& filename)
// {
//     // TODO
// }

#endif // LETTUCE_READ_BMW_FORMAT_HPP
