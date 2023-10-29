#ifndef LETTUCE_CHECKPOINT_MANAGER_HPP
#define LETTUCE_CHECKPOINT_MANAGER_HPP

// Non-standard library headers
#include "../ansi_colors.hpp"
//----------------------------------------
// Standard library headers
// ...
//----------------------------------------
// Standard C++ headers
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
//----------------------------------------
// Standard C headers
// ...

//+---------------------------------------------------------------------------------+
//| This file provides a class to save configurations automatically during runs by  |
//| using multiple rotating checkpoints. Older checkpoints will be automatically    |
//| moved to their own backup subdirectories which are named according to the       |
//| scheme 'backup1', 'backup2', ... where higher indices indicate OLDER backups.   |
//| Additionally, the class also manages loading existing checkpoints, which may be |
//| useful when restarting/extending previous runs.                                 |
//+---------------------------------------------------------------------------------+

struct CheckpointManager
{
    private:
        std::filesystem::path checkpoint_directory;
    public:
        // TODO: Make checkpoint_directory non-const?
        const int             n_checkpoint_files;

        CheckpointManager(const std::filesystem::path& checkpoint_directory_in, const int n_checkpoint_files_in, const bool create_subdirectories) noexcept :
        checkpoint_directory(checkpoint_directory_in), n_checkpoint_files(n_checkpoint_files_in)
        {
            // No need to create subdirectories when trying to read from existing directories
            if (create_subdirectories)
            {
                CreateSubdirectories();
            }
        }

        std::filesystem::path CheckpointDirectory() const
        {
            return checkpoint_directory;
        }

        void CheckpointDirectory(const std::filesystem::path& checkpoint_directory_in)
        {
            std::cout << "Changing checkpoint directory from " << checkpoint_directory << " to " << checkpoint_directory_in << std::endl;
            checkpoint_directory = checkpoint_directory_in;
        }

        std::string ReturnBackupSubdirectory(const int checkpoint_counter) const
        {
            if (checkpoint_counter == 0)
            {
                return "";
            }
            else
            {
                return "backup" + std::to_string(checkpoint_counter);
            }
        }

        void CreateSubdirectories() const
        {
            std::cout << "Creating checkpoint backup directories in " << checkpoint_directory << std::endl;
            std::error_code err;
            for (int checkpoint_count = 1; checkpoint_count < n_checkpoint_files; ++checkpoint_count)
            {
                std::filesystem::path current_backup_path {checkpoint_directory / ReturnBackupSubdirectory(checkpoint_count)};
                std::filesystem::create_directories(current_backup_path, err);
                if (err)
                {
                    std::cerr << Lettuce::Color::Red << "Creating checkpoint backup directory " << current_backup_path << " failed:\n" << err << Lettuce::Color::Reset << std::endl;
                }
            }
        }

        void RotateFiles(const std::filesystem::path& current_filepath, const std::filesystem::path& old_filepath, const bool copy_file = true) const
        {
            if (std::filesystem::exists(current_filepath))
            {
                std::error_code err;
                if (copy_file)
                {
                    std::filesystem::copy_file(current_filepath, old_filepath, std::filesystem::copy_options::overwrite_existing, err);
                }
                else
                {
                    std::filesystem::rename(current_filepath, old_filepath, err);
                }
                if (err)
                {
                    std::cerr << Lettuce::Color::Red << "Rotating file " << current_filepath << " to " << old_filepath << " failed:\n" << err << Lettuce::Color::Reset << std::endl;
                }
            }
        }

        // TODO: Currently this is a pretty redundant wrapper function. Would it make sense to move some stuff out of the SaveFunctions into this function?
        // template<typename FuncT>
        // void SaveCheckpoint(FuncT&& SaveFunction, const GaugeField& U, const std::string& filename, const bool overwrite = false) const
        // {
        //     SaveFunction(U, filename, overwrite);
        // }

        template<typename FuncT>
        void AlternatingConfigCheckpoints(FuncT&& SaveFunction, const GaugeField& U, const std::string& filename_config) const
        {
            // Go backwards from n_checkpoint_files to 1 and update the old backup checkpoints
            for (int checkpoint_counter = n_checkpoint_files - 1; checkpoint_counter > 0; --checkpoint_counter)
            {
                std::filesystem::path old_subdir     {checkpoint_directory / ReturnBackupSubdirectory(checkpoint_counter)};
                std::filesystem::path current_subdir {checkpoint_directory / ReturnBackupSubdirectory(checkpoint_counter - 1)};
                // For the last update rename/move the file so we avoid the overwrite warning in the save function below
                bool copy_file {checkpoint_counter != 1};
                RotateFiles(current_subdir / filename_config, old_subdir / filename_config, copy_file);
            }
            // For alternating checkpoints it does not make sense to disable overwrite
            constexpr bool overwrite {true};
            SaveFunction(U, checkpoint_directory / filename_config, overwrite);
        }

        template<typename FuncT, typename prngT>
        void AlternatingCheckpoints(FuncT&& SaveFunction, prngT& prng, const GaugeField& U, const std::string& filename_config, const std::string& filename_prng, const std::string& filename_normal_distribution) const
        {
            // Go backwards from n_checkpoint_files to 1 and update the old backup checkpoints
            for (int checkpoint_counter = n_checkpoint_files - 1; checkpoint_counter > 0; --checkpoint_counter)
            {
                std::filesystem::path old_subdir     {checkpoint_directory / ReturnBackupSubdirectory(checkpoint_counter)};
                std::filesystem::path current_subdir {checkpoint_directory / ReturnBackupSubdirectory(checkpoint_counter - 1)};
                // For the last update rename/move the file so we avoid the overwrite warning in the save function below
                bool copy_file {checkpoint_counter != 1};
                RotateFiles(current_subdir / filename_config,                old_subdir / filename_config,              copy_file);
                RotateFiles(current_subdir / filename_prng,                  old_subdir / filename_prng,                copy_file);
                RotateFiles(current_subdir / filename_normal_distribution,   old_subdir / filename_normal_distribution, copy_file);
            }
            // For alternating checkpoints it does not make sense to disable overwrite
            constexpr bool overwrite {true};
            SaveFunction(U, checkpoint_directory / filename_config, overwrite);
            prng.SaveState(checkpoint_directory / filename_prng, checkpoint_directory / filename_normal_distribution, overwrite);
        }

        template<typename FuncT, typename prngT>
        void LoadCheckpoint(FuncT&& LoadFunction, prngT& prng, GaugeField& U, const std::string& filename_config, const std::string& filename_prng, const std::string& filename_normal_distribution) const
        {
            LoadFunction(U, checkpoint_directory / filename_config);
            prng.LoadState(checkpoint_directory / filename_prng, checkpoint_directory / filename_normal_distribution);
        }

        void CreateCompletedFile(const std::filesystem::path& completed_filepath)
        {
            std::ofstream completed_stream(completed_filepath / "completed_run", std::fstream::out);
        }
};

#endif // LETTUCE_CHECKPOINT_MANAGER_HPP
