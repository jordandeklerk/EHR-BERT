#!/bin/bash

# Base URLs for the data files
base_urls=(
    "https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads"
    "http://downloads.cms.gov/files"
)

# List of sample numbers
samples=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

# Function to download and unzip files for a given sample number
download_and_unzip_sample() {
    sample_num=$1
    sample_dir="cms_de10_data/sample_${sample_num}"
    mkdir -p "${sample_dir}"

    files=(
        "${base_urls[0]}/de1_0_2008_beneficiary_summary_file_sample_${sample_num}.zip"
        "${base_urls[1]}/DE1_0_2008_to_2010_Carrier_Claims_Sample_${sample_num}A.zip"
        "${base_urls[1]}/DE1_0_2008_to_2010_Carrier_Claims_Sample_${sample_num}B.zip"
        "${base_urls[0]}/de1_0_2008_to_2010_inpatient_claims_sample_${sample_num}.zip"
        "${base_urls[1]}/DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_${sample_num}.zip"
        "${base_urls[0]}/de1_0_2009_beneficiary_summary_file_sample_${sample_num}.zip"
        "${base_urls[0]}/de1_0_2010_beneficiary_summary_file_sample_${sample_num}.zip"
    )

    for file in "${files[@]}"; do
        wget -P "${sample_dir}" "${file}"
        # Unzip the file if download is successful
        if [ $? -eq 0 ]; then
            unzip -o "${sample_dir}/$(basename ${file})" -d "${sample_dir}"
            rm "${sample_dir}/$(basename ${file})"
        else
            echo "Failed to download ${file}"
        fi
    done

    echo "Download and unzip complete for Sample ${sample_num}. Files saved in the '${sample_dir}' directory."
}

# Download and unzip files for each sample
for sample in "${samples[@]}"; do
    download_and_unzip_sample "${sample}"
done

echo "All downloads and unzips complete. Files saved in the 'cms_de10_data' directory."
