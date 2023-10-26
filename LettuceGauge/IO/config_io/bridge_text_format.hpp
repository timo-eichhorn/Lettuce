#ifndef LETTUCE_BRIDGE_TEXT_FORMAT_HPP
#define LETTUCE_BRIDGE_TEXT_FORMAT_HPP

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
#include <cstddef>

//+---------------------------------------------------------------------------------+
//| This file provides functions to load and save configurations stored in the      |
//| Bridge++ text format, which is structured as follows:                           |
//|- There is no header, and the links are stored as plain text                     |
//|    - Storage order is: t, z, y, x, mu (from slowest to fastest)                 |
//|    - The directions are ordered as: x, y, z, t (from slowest to fastest)        |
//|    - Each link takes up 18 lines of text (all entries are stored) and the order |
//|      of entries is:                                                             |
//|      Re(U(1, 1)) Im(U(1, 1)) Re(U(1, 2)) Im(U(1, 2)) Re(U(1, 3)) Im(U(1, 3))    |
//|      Re(U(2, 1)) Im(U(2, 1)) Re(U(2, 2)) Im(U(2, 2)) Re(U(2, 3)) Im(U(2, 3))    |
//|      Re(U(3, 1)) Im(U(3, 1)) Re(U(3, 2)) Im(U(3, 2)) Re(U(3, 3)) Im(U(3, 3))    |
//+---------------------------------------------------------------------------------+

// bool CheckFormatBridgeText(const std::string& filename)
// {
//     // TODO: Implement
// }

bool LoadConfigBridgeText(GaugeField& U, const std::string& filename)
{
    // TODO: Reorder for better alignment?
    std::ifstream         config_stream;

    // Attempt to open file and check stream status
    config_stream.open(filename, std::fstream::in);
    if (config_stream.fail())
    {
        std::cout << Lettuce::Color::BoldRed << "Error while opening file " << filename << "!" << Lettuce::Color::Reset << std::endl;
        return false;
    }

    // TODO: This could be a generic function used for all formats
    std::cout << Lettuce::Color::BoldBlue << "Attempting to read configuration in Bridge++ text format from " << filename << ":" << Lettuce::Color::Reset << std::endl;
    if (!std::filesystem::exists(filename))
    {
        std::cout << Lettuce::Color::BoldRed << "File " << filename << " not found!" << Lettuce::Color::Reset << std::endl;
        return false;
    }

    // Count the total linenumber using std::count (add one to the final result since we only counted the number of '\n' characters)
    std::size_t linecount = std::count(std::istreambuf_iterator<char>(config_stream), std::istreambuf_iterator<char>(), '\n') + 1;
    // Check if number of lines in file is compatible with current lattice volume (unfortunately we can not check the lengths since the Bridge++ text format does not contain any information about the exact shape)
    bool lengths_match {U.Volume() * 4 * 18 == linecount};
    std::string indent_whitespace {"    "};
    if (not lengths_match)
    {
        std::cout << Lettuce::Color::BoldRed << indent_whitespace << "Lattice shapes do not match!\n";
        std::cout                            << indent_whitespace << "Expected number of lines: " << U.Volume() * 4 * 18 << "\n";
        std::cout                            << indent_whitespace << "Number of lines in file: " << linecount << Lettuce::Color::Reset << std::endl;
        return false;
    }
    // Return to beginning of file
    config_stream.clear();
    config_stream.seekg(0, config_stream.beg);
    std::cout << Lettuce::Color::BoldBlue << indent_whitespace << "Reading lattice from " << filename << "..." << Lettuce::Color::Reset << std::endl;

    auto start_read_config {std::chrono::high_resolution_clock::now()};
    // Bridge++ text format link order corresponds to array[Nt][Nz][Ny][Nx][Nmu], i.e., t is the slowest and mu the fastest index
    for (int t = 0; t < Nt; ++t)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y)
    for (int x = 0; x < Nx; ++x)
    for (auto mu : {1, 2, 3, 0})
    {
        link_coord current_link {t, x, y, z, mu};
        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            double re_part, im_part;
            config_stream >> re_part;
            config_stream >> im_part;
            U(current_link)(a, b) = std::complex(re_part, im_part);
        }
    }
    config_stream.close();
    config_stream.clear();
    // std::cout << indent_whitespace << "Checksum (from file):    " << checksum_read_string << std::endl;
    // std::cout << indent_whitespace << "Checksum (recalculated): " << checksum_new.ReturnString() << Lettuce::Color::Reset << std::endl;
    auto end_read_config {std::chrono::high_resolution_clock::now()};
    // std::chrono::duration<double> read_time_header {end_read_header - start_read_header};
    std::chrono::duration<double> read_time_config {end_read_config - start_read_config};
    // std::cout << indent_whitespace << "Time for reading header: " << read_time_header.count() << "\n";
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

bool SaveConfigBridgeText(const GaugeField& U, const std::string& filename, const bool overwrite = false)
{
    // If file already exists, abort
    // TODO: This could be a generic function used for all formats
    std::cout << Lettuce::Color::BoldBlue << "Attempting to write configuration in Bridge++ text format to " << filename << ":" << Lettuce::Color::Reset << std::endl;
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
            std::cerr << Lettuce::Color::BoldRed << "Writing configuration in Bridge++ text format to file " << filename << " failed!" << Lettuce::Color::Reset << std::endl;
            return false;
        }
    }

    std::ofstream config_ofstream;
    // TODO: Check what kind of format the Bridge++ code actually uses
    config_ofstream << std::setprecision(16) << std::fixed;
    // config_ofstream.open(filename, std::fstream::out);
    config_ofstream.open(filename, std::ios::trunc);
    // config_ofstream.setf(std::ios_base::scientific, std::ios_base::floatfield);
    // config_ofstream.precision(14);

    // Write links
    auto start_write_config {std::chrono::high_resolution_clock::now()};
    std::array<double, 12> buffer;
    for (int t = 0; t < Nt; ++t)
    for (int z = 0; z < Nz; ++z)
    for (int y = 0; y < Ny; ++y)
    for (int x = 0; x < Nx; ++x)
    for (auto mu : {1, 2, 3, 0})
    {
        link_coord current_link {t, x, y, z, mu};
        for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            std::complex entry {U({t, x, y, z, mu})(a, b)};
            config_ofstream << std::real(entry) << "\n" << std::imag(entry) << "\n";
        }

    }
    auto end_write_config {std::chrono::high_resolution_clock::now()};
    if (!config_ofstream)
    {
        std::cerr << Lettuce::Color::BoldRed << "Writing config to file " << filename << " failed!" << Lettuce::Color::BoldRed << std::endl;
        return false;
    }
    // std::chrono::duration<double> write_time_header {end_write_header - start_write_header};
    std::chrono::duration<double> write_time_config {end_write_config - start_write_config};
    // std::cout << indent_whitespace << "Time for writing header: " << write_time_header.count() << "\n";
    std::cout << indent_whitespace << "Time for writing config: " << write_time_config.count() << std::endl;
    return true;
}

#endif // LETTUCE_BRIDGE_TEXT_FORMAT_HPP
