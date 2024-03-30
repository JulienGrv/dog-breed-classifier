#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_results.py
#
# PROGRAMMER: Julien Grave
# DATE CREATED: 28.03.24
# REVISED DATE: 30.03.24
# PURPOSE: Create a function print_results that prints the results statistics
#          from the results statistics dictionary (results_stats_dic). It
#          should also allow the user to be able to print out cases of misclassified
#          dogs and cases of misclassified breeds of dog using the Results
#          dictionary (results_dic).
#         This function inputs:
#            -The results dictionary as results_dic within print_results
#             function and results for the function call within main.
#            -The results statistics dictionary as results_stats_dic within
#             print_results function and results_stats for the function call within main.
#            -The CNN model architecture as model within print_results function
#             and in_arg.arch for the function call within main.
#            -Prints Incorrectly Classified Dogs as print_incorrect_dogs within
#             print_results function and set as either boolean value True or
#             False in the function call within main (defaults to False)
#            -Prints Incorrectly Classified Breeds as print_incorrect_breed within
#             print_results function and set as either boolean value True or
#             False in the function call within main (defaults to False)
#         This function does not output anything other than printing a summary
#         of the final results.
##

from typing import Any, Literal

from calculates_results_stats import Stats


def print_results(
    results_dic: dict[str, list[Any]],
    results_stats_dic: Stats,
    model: Literal["alexnet", "resnet", "vgg"],
    print_incorrect_dogs: bool = False,
    print_incorrect_breed: bool = False,
) -> None:
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifier labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats_dic - Dictionary that contains the results statistics (either
                   a  percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - Indicates which CNN model architecture will be used by the
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """
    print(f"\n\n*** Results Summary for CNN Model Architecture {model} ***")
    print(f"{'N Images':20}: {results_stats_dic['n_images']:3d}")
    print(f"{'N Dog Images':20}: {results_stats_dic['n_dogs_img']:3d}")
    print(f"{'N Not-Dog Images':20}: {results_stats_dic['n_notdogs_img']:3d}")

    for key, value in results_stats_dic.items():
        if key.startswith("pct_"):
            print(f"{key:20}: {value:3.2f}%")

    if print_incorrect_dogs and (
        results_stats_dic["n_correct_dogs"] + results_stats_dic["n_correct_notdogs"]
        != results_stats_dic["n_images"]
    ):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        print(
            "\n".join(
                f"Real: {result[0]:>26}   Classifier: {result[1]:>30}"
                for result in results_dic.values()
                if result[3] + result[4] == 1
            )
        )

    if (
        print_incorrect_breed
        and results_stats_dic["n_correct_dogs"] != results_stats_dic["n_correct_breed"]
    ):
        print("\nINCORRECT Dog Breed Assignments:")
        print(
            "\n".join(
                f"Real: {result[0]:>26}   Classifier: {result[1]:>30}"
                for result in results_dic.values()
                if result[2] == 0 and result[3] == 1 and result[4] == 1
            )
        )
