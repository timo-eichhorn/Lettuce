#ifndef LETTUCE_BMW_FORMAT_HPP
#define LETTUCE_BMW_FORMAT_HPP

// Non-standard library headers
#include "../../math/su3.hpp"
#include "../ansi_colors.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <complex>
#include <filesystem>
#include <ios>
#include <string>
// #include <fstream>
//----------------------------------------
// Standard C headers
#include <cstdio>

//+---------------------------------------------------------------------------------+
//| This file provides functions to load and save configurations stored in the BMW  |
//| format, which is structured as follows:                                         |
//|- All data is encoded in ASCII                                                   |
//|- All configuration files begin with a header of length 4096 bytes               |
//|    - The first header line reads: #BMW <Nx> <Ny> <Nz> <Nt> <HEX checksum>       |
//|    - The rest of the header may be used freely (e.g. name of the code used to   |
//|      generate the configuration, other parameters, average plaquette, ...), but |
//|      by default empty characters are filled with "\n"                           |
//|    - The checksum is a 64-bit Adler checksum calculated from the links/non-     |
//|      header part of the file                                                    |
//|- After the header, the links follow                                             |
//|    - Storage order is: t, z, y, x, mu (from slowest to fastest)                 |
//|    - The directions are ordered as: x, y, z, t (from slowest to fastest)        |
//|    - Each link takes up 96 bytes (first two rows in double precision), and the  |
//|      order of entries is:                                                       |
//|      Re(U(1, 1)) Im(U(1, 1)) Re(U(1, 2)) Im(U(1, 2)) Re(U(1, 3)) Im(U(1, 3))    |
//|      Re(U(2, 1)) Im(U(2, 1)) Re(U(2, 2)) Im(U(2, 2)) Re(U(2, 3)) Im(U(2, 3))    |
//|    - The entries are stored in big-endian format                                |
//+---------------------------------------------------------------------------------+

// TODO: Lot's of cleanup still left to do here...
//       Also benchmark how different buffer sizes (12 vs N_mu * 12) affect the performance

// bool CheckFormatBMW(const std::string& filename)
// {
//     // TODO: Implement
// }

struct Checksum_Adler64
{
    // Largest prime smaller than 2^32
    inline static constexpr std::uint32_t Adler64_prime {4294967291U};
    std::uint64_t                         final         {Adler64_prime};
    std::uint64_t                         A             {0};
    std::uint64_t                         B             {0};

    void Add(std::uint64_t* data, std::uint64_t block_id, std::uint64_t number_of_blocks_in_total, int block_size_in_64bit_units) noexcept
    {
        std::uint64_t sum_a {0};
        std::uint64_t sum_b {0};
        std::uint64_t revi;
        revi = (number_of_blocks_in_total - block_id) * static_cast<unsigned int>(2 * block_size_in_64bit_units);
        for(int i = 0; i < block_size_in_64bit_units; ++i)
        {
            std::uint64_t d0 {data[i] >> 32};
            std::uint64_t d1 {data[i] & 0xffffffff};
            sum_a += d0 + d1;
            sum_b += (2 + revi * d0 + (revi - 1) * d1) % Adler64_prime;
            revi  -= 2;
        }
        A = (A + sum_a) % Adler64_prime;
        B = (B + sum_b) % Adler64_prime;
    }

    void Finalize() noexcept
    {
        // Introduce copies of A and B so that repeated calls to Finalize() do not change the result
        std::uint64_t A_copy {(A + 1) % Adler64_prime};
        std::uint64_t B_copy {B % Adler64_prime};
        // TODO: In Szabolcs' code, there is a call here to kernel_globsum_uint64, which is an empty function with the note "no communication: do nothing"?
        final = (B_copy << 32) + A_copy;
    }

    std::string ReturnString() noexcept
    {
        if (final == Adler64_prime)
        {
            std::cout << Lettuce::Color::BoldRed << "Invalid value for final in Checksum_Adler64 struct! Make sure to call Finalize() first!" << Lettuce::Color::Reset << std::endl;
            // TODO: Perhaps better an exception here?
            // return std::to_string(-1);
        }
        // return std::sprintf(target, "%016jx", static_cast<std::uintmax_t>(final));
        // TODO: Instead of stringstreams use std::format once supported by most compilers?
        std::stringstream hex_stream;
        hex_stream << std::hex << final;
        return hex_stream.str();
    }
};



template<typename T>
void SwapEndianness(T* in) noexcept
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

std::array<double, 12> DeconstructMatBMW(const Matrix_SU3& link) noexcept
{
    // BMW format stores only the first two rows
    return {std::real(link(0, 0)), std::imag(link(0, 0)), std::real(link(0, 1)), std::imag(link(0, 1)), std::real(link(0, 2)), std::imag(link(0, 2)),
            std::real(link(1, 0)), std::imag(link(1, 0)), std::real(link(1, 1)), std::imag(link(1, 1)), std::real(link(1, 2)), std::imag(link(1, 2))};

}

bool LoadConfigBMW(GaugeField& U, const std::string& filename)
{
    // TODO: Reorder for better alignment?
    // std::ifstream         config_stream;
    const std::size_t     header_block_size {4096};
    const std::size_t     object_size {1};
    char                  header_block[header_block_size];
    site_coord            lattice_lengths {0, 0, 0, 0};
    char                  checksum_read_tmp[32];
    // char                  checksum_new[32];
    Checksum_Adler64      checksum_new;
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
    if (!std::filesystem::exists(filename))
    {
        std::cout << Lettuce::Color::BoldRed << "File " << filename << "not found!" << Lettuce::Color::Reset << std::endl;
        return false;
    }

    const char* filename_cstr   = filename.c_str();
    std::FILE*  file            = std::fopen(filename_cstr, "r");
    std::string indent_whitespace {"    "};
    auto        start_read_header {std::chrono::high_resolution_clock::now()};

    // Read header block and check if file matches expected format
    // config_stream.read(reinterpret_cast<char*>(), header_block_size)
    std::size_t successful_reads   = std::fread(header_block, object_size, header_block_size, file);
    // BMW format stores the lattice lenghts in the following order: (x, y, z, t)
    std::size_t successful_assigns = std::sscanf(header_block, "#BMW %d %d %d %d %31s %n", &lattice_lengths.x, &lattice_lengths.y, &lattice_lengths.z, &lattice_lengths.t, checksum_read_tmp, &header_characters_read);
    if (successful_reads != header_block_size or successful_assigns < 5)
    {
        std::cout << Lettuce::Color::BoldRed << indent_whitespace << "Header block does not match expected format!" << Lettuce::Color::Reset << std::endl;
        return false;
    }

    // Check if lattice shape in file matches the current lattice shape
    bool lengths_match {lattice_lengths == U.Shape()};
    if (not lengths_match)
    {
        std::cout << Lettuce::Color::BoldRed << indent_whitespace << "Lattice shapes do not match!\n";
        std::cout                            << indent_whitespace << "Current shape: " << U.Shape() << "\n";
        std::cout                            << indent_whitespace << "Shape in file: " << lattice_lengths << Lettuce::Color::Reset << std::endl;
        return false;
    }
    std::cout << Lettuce::Color::BoldBlue << indent_whitespace << "Reading lattice from " << filename << "..." << Lettuce::Color::Reset << std::endl;
    auto end_read_header {std::chrono::high_resolution_clock::now()};

    auto start_read_config {std::chrono::high_resolution_clock::now()};
    // std::array<double, 4 * 12> buffer;
    // TODO: Array of array might cause problems since there can be padding between the arrays... (problematic below where we add to checksum)
    std::array<std::array<double, 12>, 4> buffer;
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
    //         SwapEndianness(buffer + i);
    //     }
    //     U({t, x, y, z, 1}) = ReconstructMatBMW(buffer);
    //     U({t, x, y, z, 2}) = ReconstructMatBMW(buffer + 12);
    //     U({t, x, y, z, 3}) = ReconstructMatBMW(buffer + 24);
    //     U({t, x, y, z, 0}) = ReconstructMatBMW(buffer + 36);
    // }

    // for (auto mu : {1, 2, 3, 0})
    // {
    //     std::fread(buffer.data(), sizeof(double), 12, file);
    //     for (int i = 0; i < 12; ++i)
    //     {
    //         SwapEndianness(buffer.data() + i);
    //     }
    //     std::uint64_t site_abs_value {static_cast<std::uint64_t>(((x * Ny + y) * Nz + z) * Nt + t)};
    //     checksum_new.Add(reinterpret_cast<std::uint64_t*>(buffer.data()), site_abs_value, U.Volume(), 4 * 12);
    //     U({t, x, y, z, mu}) = ReconstructMatBMW(buffer);
    // }
    {
        site_coord current_site {t, x, y, z};
        for (auto mu : {1, 2, 3, 0})
        {
            int mu_permuted {(mu + 3) % 4};
            // int memory_offset {mu_permuted * 12};
            std::fread(buffer[mu_permuted].data(), sizeof(double), 12, file);
            for (int i = 0; i < 12; ++i)
            {
                SwapEndianness(buffer[mu_permuted].data() + i);
            }
            U({t, x, y, z, mu}) = ReconstructMatBMW(buffer[mu_permuted]);
        }
        std::uint64_t site_abs_value {static_cast<std::uint64_t>(((current_site.t * Nz + current_site.z) * Ny + current_site.y) * Nx + current_site.x)};
        checksum_new.Add(reinterpret_cast<std::uint64_t*>(buffer.data()), site_abs_value, U.Volume(), 4 * 12);
    }
    checksum_new.Finalize();
    std::string checksum_read_string = checksum_read_tmp;
    if (checksum_read_string == checksum_new.ReturnString())
    {
        std::cout << indent_whitespace << Lettuce::Color::BoldGreen << "Checksums match!" << std::endl;
    }
    else
    {
        std::cout << indent_whitespace << Lettuce::Color::BoldRed << "Checksums do not match!" << std::endl;
    }
    std::cout << indent_whitespace << "Checksum (from file):    " << checksum_read_string << std::endl;
    std::cout << indent_whitespace << "Checksum (recalculated): " << checksum_new.ReturnString() << Lettuce::Color::Reset << std::endl;
    auto end_read_config {std::chrono::system_clock::now()};
    std::chrono::duration<double> read_time_header {end_read_header - start_read_header};
    std::chrono::duration<double> read_time_config {end_read_config - start_read_config};
    std::cout << indent_whitespace << "Time for reading header: " << read_time_header.count() << "\n";
    std::cout << indent_whitespace << "Time for reading config: " << read_time_config.count() << std::endl;

    bool InGroup {SU3::Tests::IsGroupElement(U)};
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

bool SaveConfigBMW(const GaugeField& U, const std::string& filename, const bool overwrite = false)
{
    // If file already exists, abort
    std::cout << Lettuce::Color::BoldBlue << "Attempting to write configuration in BMW format to " << filename << ":" << Lettuce::Color::Reset << std::endl;
    const std::string indent_whitespace {"    "};
    if (std::filesystem::exists(filename))
    {
        std::cout << indent_whitespace << Lettuce::Color::BoldRed << "File " << filename << " already exists!" << Lettuce::Color::Reset << std::endl;
        if (overwrite)
        {
            std::cout << indent_whitespace << Lettuce::Color::BoldRed << "Overwriting existing file..." << Lettuce::Color::Reset << std::endl;
        }
        else
        {
            return false;
        }
    }

    std::ofstream config_ofstream;
    config_ofstream.open(filename, std::ios::out | std::ios::binary);
    const std::size_t     header_block_size {4096};
    char                  header_block[header_block_size];
    int                   header_offset {0};
    Checksum_Adler64      checksum;

    // Write header
    auto start_write_header {std::chrono::high_resolution_clock::now()};
    // Calculate checksum
    for (int t = 0; t < Nt; ++t)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y)
    for (int x = 0; x < Nx; ++x)
    {
        std::uint64_t buffer[4 * 12];
        site_coord current_site {t, x, y, z};
        // Direction order: x, y, z, t
        for (auto mu : {1, 2, 3, 0})
        {
            // BMW order:     x, y, z, t
            // Lettuce order: t, x, y, z
            // Get the matching BMW direction index which we shall call mu_permuted
            int mu_permuted {(mu + 3) % 4};
            // Reunitarize/project copy of link
            Matrix_SU3 tmp {U(current_site, mu)};
            // SU3::Projection::GramSchmidt(tmp);
            // TODO: If we use the function above, we get different results than with Szabolcs' code
            // In Szabolcs' code, the projection looks as follows
            // #define rdot(A,B) (creal((A)[0])*creal((B)[0])+cimag((A)[0])*cimag((B)[0])+ creal((A)[1])*creal((B)[1])+cimag((A)[1])*cimag((B)[1])+ creal((A)[2])*creal((B)[2])+cimag((A)[2])*cimag((B)[2]))
            // #define cdot(A,B) (A)[0]*conj((B)[0])+(A)[1]*conj((B)[1])+(A)[2]*conj((B)[2])
            // double n;
            // complex double c;
            // n=rdot(U->c[0],U->c[0]);
            // n=1./sqrt(n);
            // U->c[0][0]*=n; 
            // U->c[0][1]*=n;
            // U->c[0][2]*=n;
            // c=cdot(U->c[1],U->c[0]);
            // U->c[1][0]-=c*U->c[0][0];
            // U->c[1][1]-=c*U->c[0][1];
            // U->c[1][2]-=c*U->c[0][2];
            // n=rdot(U->c[1],U->c[1]);
            // n=1.0/sqrt(n);
            // U->c[1][0]*=n;
            // U->c[1][1]*=n;
            // U->c[1][2]*=n;
            // std::memcpy(buffer + mu_permuted * 12, DeconstructMatBMW(tmp).data(), 96);
            std::memcpy(buffer + mu_permuted * 12, DeconstructMatBMW(tmp).data(), 96);
        }
        std::uint64_t site_abs_value {static_cast<std::uint64_t>(((current_site.t * Nz + current_site.z) * Ny + current_site.y) * Nx + current_site.x)};
        checksum.Add(buffer, site_abs_value, U.Volume(), 4 * 12);
    }
    checksum.Finalize();
    std::string checksum_string {checksum.ReturnString()};
    std::cout << indent_whitespace << "Checksum: " << checksum_string << std::endl;
    // Fill header with newline characters
    std::memset(header_block, '\n', header_block_size);
    // First header line
    site_coord lattice_lengths {U.Shape()};
    header_offset = std::sprintf(header_block, "#BMW %d %d %d %d %s\n", lattice_lengths[1], lattice_lengths[2], lattice_lengths[3], lattice_lengths[0], checksum_string.c_str());
    // Additional header comments
    // TODO: Insert git hash here
    header_offset += std::sprintf(header_block + header_offset, "\nGenerated with Lettuce\n");
    // TODO: Check if header length is exceeded (perhaps better if we work with a string here instead of a char array?)
    // Write header to file
    config_ofstream.write(header_block, header_block_size);
    auto end_write_header {std::chrono::high_resolution_clock::now()};

    // Write links
    auto start_write_config {std::chrono::high_resolution_clock::now()};
    std::array<double, 12> buffer;
    for (int t = 0; t < Nt; ++t)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y)
    for (int x = 0; x < Nx; ++x)
    for (auto mu : {1, 2, 3, 0})
    {
        Matrix_SU3 tmp {U({t, x, y, z, mu})};
        // SU3::Projection::GramSchmidt(tmp);
        buffer = DeconstructMatBMW(tmp);
        // Swap to big endian (assuming the host machine is little endian)
        for (int i = 0; i < 12; ++i)
        {
            SwapEndianness(buffer.data() + i);
        }
        config_ofstream.write(reinterpret_cast<const char*>(buffer.data()), sizeof(buffer));
    }
    config_ofstream.close();
    config_ofstream.clear();
    auto end_write_config {std::chrono::system_clock::now()};
    std::chrono::duration<double> write_time_header {end_write_header - start_write_header};
    std::chrono::duration<double> write_time_config {end_write_config - start_write_config};
    std::cout << indent_whitespace << "Time for writing header: " << write_time_header.count() << "\n";
    std::cout << indent_whitespace << "Time for writing config: " << write_time_config.count() << std::endl;
    return true;
}

#endif // LETTUCE_BMW_FORMAT_HPP
