#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include "habituating_neuron.hpp"

int main() {
    std::ifstream input("ECG5000/ECG5000_TEST.ts");
    std::ofstream output("results/ecg_multi_neuron_output.csv");

    if (!input.is_open() || !output.is_open()) {
        std::cerr << "Failed to open input/output files.\n";
        return 1;
    }

    std::string line;
    bool in_data_section = false;
    int sequence_id = 0;

    // Write CSV header
    output << "sequence_id,label,timestep,value,"
           << "response1,novelty1,"
           << "response2,novelty2,"
           << "response3,novelty3\n";

    while (std::getline(input, line)) {
        if (line.empty()) continue;

        if (line == "@data") {
            in_data_section = true;
            continue;
        }

        if (!in_data_section) continue;

        std::stringstream ss(line);
        std::string item;
        std::vector<float> values;

        // Parse label
        std::getline(ss, item, ',');
        int label = std::stoi(item);  // Class label (1 = normal)

        // Parse ECG values
        while (std::getline(ss, item, ',')) {
            values.push_back(std::stof(item));
        }

        // === Adaptive Threshold Calculation ===
        double sum = 0.0, sq_sum = 0.0;
        for (float v : values) {
            sum += v;
            sq_sum += v * v;
        }
        double mean = sum / values.size();
        double variance = (sq_sum / values.size()) - (mean * mean);
        double stddev = std::sqrt(variance);
        double adaptive_threshold = mean + 0.5 * stddev;

        // Initialize 3 neurons with different parameters
        HabituatingNeuron neuron1(0.05, 0.2, 0.3);  // Balanced
        HabituatingNeuron neuron2(0.03, 0.1, 0.25); // Slow to habituate
        HabituatingNeuron neuron3(0.07, 0.3, 0.4);  // Fast and sensitive

        // Feed sequence into all neurons
        for (size_t t = 0; t < values.size(); ++t) {
            int stimulus = (values[t] > adaptive_threshold) ? 1 : 0;

            bool is_novel1 = neuron1.update(stimulus);
            bool is_novel2 = neuron2.update(stimulus);
            bool is_novel3 = neuron3.update(stimulus);

            output << sequence_id << "," << label << "," << t << "," << values[t] << ","
                   << neuron1.get_response() << "," << is_novel1 << ","
                   << neuron2.get_response() << "," << is_novel2 << ","
                   << neuron3.get_response() << "," << is_novel3 << "\n";
        }

        sequence_id++;
    }

    input.close();
    output.close();

    std::cout << "Multi-neuron simulation complete. Output written to results/ecg_multi_neuron_output.csv\n";
    return 0;
}