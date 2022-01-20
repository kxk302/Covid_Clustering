import argparse
import json

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input_file', required=True)
  parser.add_argument('-a', '--study_accessions_file', required=True)
  parser.add_argument('-s', '--samples_file', required=True)
  args = parser.parse_args()
  
  study_accessions = None
  input = None
  samples = []

  with open(args.input_file, 'r') as f:
    input = json.load(f) 

  with open(args.study_accessions_file, 'r') as f:
    study_accessions = f.read().splitlines() 

  print(study_accessions)
  print(type(study_accessions))
  print(len(study_accessions))

  print(type(input))
  print(len(input))
 
  total_sa_count = 0
  for study_accession in study_accessions:
    sa_count = 0
    for (key, value) in input.items():
      if value['study_accession'] == study_accession:
        samples.extend(value['samples'])
        print(len(value['samples']))
        sa_count += len(value['samples'])
    print('Number of samples for ' + study_accession + ': ' + str(sa_count))
    total_sa_count += sa_count

  print(samples)
  print(len(samples))
  print('total_sa_count: ' + str(total_sa_count))

  with open(args.samples_file, 'w') as f:
    for sample in samples:
      f.write(sample+'\n')
