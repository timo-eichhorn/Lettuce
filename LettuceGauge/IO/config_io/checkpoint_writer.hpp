#ifndef LETTUCE_CHECKPOINT_WRITER_HPP
#define LETTUCE_CHECKPOINT_WRITER_HPP

// Non-standard library headers
// ...
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
// #include <string_view>
//----------------------------------------
// Standard C headers
// ...

//+---------------------------------------------------------------------------------+
//| This file provides a class to save configurations automatically during runs by  |
//| using multiple rotating checkpoints.                                            |
//+---------------------------------------------------------------------------------+

struct CheckpointWriter
{
    public:
        const int n_checkpoint_files;
    // private:
    //           int checkpoint_counter {0};
    public:
        CheckpointWriter(const int n_checkpoint_files_in) noexcept :
        n_checkpoint_files(n_checkpoint_files_in)
        {}

        std::string ReturnCheckpointAppendix(const int checkpoint_counter) const
        {
            if (checkpoint_counter == 0)
            {
                return "";
            }
            else
            {
                return ".bak" + std::to_string(checkpoint_counter);
            }
        }

        void RotateFiles(std::string_view current_filename, std::string_view old_filename) const
        {
            std::cout << "Old file:     " << old_filename << std::endl;
            std::cout << "Current file: " << current_filename << std::endl;
            if (std::filesystem::exists(current_filename))
            {
                // std::filesystem::copy_file(current_filename, old_filename, std::filesystem::copy_options::overwrite_existing);
                std::cout << "Copying file " << current_filename << " to file " << old_filename << std::endl;
                bool tmp = std::filesystem::copy_file(current_filename, old_filename, std::filesystem::copy_options::overwrite_existing);
                std::cout << "Success status: " << tmp << std::endl;
            }
        }

        // TODO: Currently this is a pretty redundant wrapper function. Would it make sense to move some stuff out of the SaveFunctions into this function?
        template<typename FuncT>
        void SaveCheckpoint(FuncT&& SaveFunction, const GaugeField& U, const std::string& filename, const bool overwrite = false) const
        {
            SaveFunction(U, filename, overwrite);
        }

        // The most up-to-date checkpoint is always saved as 'filename', while backups are saved as 'filename.bak1', 'filename.bak2', ...
        // Note that here the highest number indicates the file is the OLDEST
        template<typename FuncT>
        void AlternatingConfigCheckpoints(FuncT&& SaveFunction, const GaugeField& U, const std::string& filename_config) const
        {
            // For alternating checkpoints it does not make sense to disable overwrite
            constexpr bool overwrite {true};
            // Go backwards from n_checkpoint_files to 1 and update the old backup checkpoints
            // If the file in question exists rename it by changing the index from n to n + 1
            // In the case that n = n_checkpoint_files delete the file
            for (int checkpoint_counter = n_checkpoint_files - 1; checkpoint_counter > 0; --checkpoint_counter)
            {
                std::string old_file_appendix     {ReturnCheckpointAppendix(checkpoint_counter)};
                std::string current_file_appendix {ReturnCheckpointAppendix(checkpoint_counter - 1)};
                RotateFiles(filename_config + current_file_appendix, filename_config + old_file_appendix);
            }
            SaveFunction(U, filename_config, overwrite);
        }

        // The most up-to-date checkpoint is always saved as 'filename', while backups are saved as 'filename.bak1', 'filename.bak2', ...
        // Note that here the highest number indicates the file is the OLDEST
        template<typename FuncT, typename prngT>
        void AlternatingCheckpoints(FuncT&& SaveFunction, prngT& prng, const GaugeField& U, const std::string& filename_config, const std::string& filename_prng, const std::string& filename_normal_distribution) const
        {
            // For alternating checkpoints it does not make sense to disable overwrite
            constexpr bool overwrite {true};
            // Go backwards from n_checkpoint_files to 1 and update the old backup checkpoints
            // If the file in question exists rename it by changing the index from n to n + 1
            // In the case that n = n_checkpoint_files delete the file
            for (int checkpoint_counter = n_checkpoint_files - 1; checkpoint_counter > 0; --checkpoint_counter)
            {
                std::string old_file_appendix     {ReturnCheckpointAppendix(checkpoint_counter)};
                std::string current_file_appendix {ReturnCheckpointAppendix(checkpoint_counter - 1)};
                RotateFiles(filename_config              + current_file_appendix, filename_config              + old_file_appendix);
                RotateFiles(filename_prng                + current_file_appendix, filename_prng                + old_file_appendix);
                RotateFiles(filename_normal_distribution + current_file_appendix, filename_normal_distribution + old_file_appendix);
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            }
            SaveFunction(U, filename_config, overwrite);
            prng.SaveState(filename_prng, filename_normal_distribution, overwrite);
        }
};

#endif // LETTUCE_CHECKPOINT_WRITER_HPP
