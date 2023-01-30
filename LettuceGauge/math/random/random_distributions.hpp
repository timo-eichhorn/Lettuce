#ifndef LETTUCE_RANDOM_DISTRIBUTIONS_HPP
#define LETTUCE_RANDOM_DISTRIBUTIONS_HPP

// Non-standard library headers
#include <../../defines.hpp>
//----------------------------------------
// Standard library headers
#include <limits>
//----------------------------------------
// Standard C++ headers
// ...
//----------------------------------------
// Standard C headers
#include <cmath>

namespace Lettuce
{
    template<typename RealType = double>
    class NormalDistribution
    {
        public:
            using result_type = RealType;

            struct param_type
            {
                private:
                    RealType mean_;
                    RealType stddev_;
                public:
                    using distribution_type = NormalDistribution<RealType>;

                    param_type() noexcept :
                    param_type(0.0)
                    {}

                    explicit param_type(RealType mean_in, RealType stddev_in = static_cast<RealType>(1)) noexcept :
                    mean_(mean_in), stddev_(stddev_in)
                    {
                        assert(stddev_ > static_cast<RealType>(0));
                    }

                    [[nodiscard]]
                    RealType mean() const noexcept
                    {
                        return mean_;
                    }

                    [[nodiscard]]
                    RealType stddev() const noexcept
                    {
                        return stddev_;
                    }

                    friend bool operator==(const param_type& param1, const param_type& param2)
                    {
                        return (param1.mean() == param2.mean() and param1.stddev() == param2.stddev());
                    }

                    friend bool operator!=(const param_type& param1, const param_type& param2)
                    {
                        return !(param1 == param2);
                    }
            };

            NormalDistribution() noexcept :
            NormalDistribution(0.0)
            {}

            explicit NormalDistribution(RealType mean_in, RealType stddev_in = static_cast<RealType>(1)) noexcept :
            parameter_(mean_in, stddev_in)
            {}

            explicit NormalDistribution(const param_type& parameter_in) noexcept :
            parameter_(parameter_in)
            {}

            [[nodiscard]]
            RealType mean() const noexcept
            {
                return parameter_.mean();
            }

            [[nodiscard]]
            RealType stddev() const noexcept
            {
                return parameter_.stddev();
            }

            [[nodiscard]]
            param_type param() const noexcept
            {
                return parameter_;
            }

            void param(const param_type& parameter_in)
            {
                parameter_ = parameter_in;
            }

            result_type min() const
            {
                return std::numeric_limits<result_type>::lowest();
            }

            result_type max() const
            {
                return std::numeric_limits<result_type>::max();
            }

            template<typename prngT>
            result_type operator()(prngT& prng)
            {
                result_type tmp;
                if (saved_available_)
                {
                    tmp = saved_;
                    saved_available_ = false;
                }
                else
                {
                    // Marsaglia polar method (same method as used in libstdc++)
                    result_type x, y, r_sq;
                    do
                    {
                        x = static_cast<result_type>(2.0) * prng() - static_cast<result_type>(1.0);
                        y = static_cast<result_type>(2.0) * prng() - static_cast<result_type>(1.0);
                        r_sq = x * x + y * y;
                    }
                    while (r_sq >= static_cast<result_type>(1.0) or r_sq == static_cast<result_type>(0.0));

                    result_type factor {std::sqrt(-2 * std::log(r_sq) / r_sq)};
                    saved = x * factor;
                    saved_available_ = true;
                    tmp   = y * factor;
                }
                return tmp * parameter_.stddev() + parameter_.mean();
            }
        private:
            param_type  parameter_;
            result_type saved_;
            bool        saved_available_ = false;
    };

} // namespace Lettuce

#endif // LETTUCE_RANDOM_DISTRIBUTIONS_HPP
