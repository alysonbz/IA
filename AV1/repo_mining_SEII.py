from pydriller import Repository
from collections import Counter
from datetime import datetime

import pydriller
print("1. Qual a quantidade de commmit do projeto")
url_junit = "https://github.com/junit-team/junit5.git"

cont = 0
for commit in Repository(url_junit).traverse_commits():
    cont += 1
print(cont)

print("2. Quantos commits foream realizados em 2022? Em 2023? Em 2024")
dt1 = datetime(2022, 1, 1, 17, 0, 0)
dt2 = datetime(2024, 12, 30, 17, 59, 0)

cont2022 = 0
cont2023 = 0
cont2024 = 0
for commit in Repository(url_junit , since=dt1, to=dt2).traverse_commits():
  if(commit.author_date.year == 2022):
    cont2022 += 1
  if(commit.author_date.year == 2023):
    cont2023 += 1
  if(commit.author_date.year == 2024):
    cont2024 += 1

print("Em 2022 foram: ", cont2022)
print("Em 2023 foram: ", cont2022)
print("Em 2024 foram: ", cont2024)

print(" 3. Quantos commits incluem a string “feature” na mensagem do commit? E a string “fix”?")
feature_count = 0
fix_count = 0
for commit in Repository(url_junit).traverse_commits():
    if 'feature' in commit.msg.lower():
        feature_count += 1

    # Verifica se "fix" está na mensagem do commit
    if 'fix' in commit.msg.lower():
        fix_count += 1

print(f'Commits com "feature": {feature_count}')
print(f'Commits com "fix": {fix_count}')

print("4. Quais os 5 arquivos “.java” ou / “.js” ou / “.py” mais modificados?")
file_modifications = Counter()

for commit in Repository(url_junit).traverse_commits():
    for modified_file in commit.modified_files:
        if modified_file.filename.endswith(('.java', '.js', '.py')):
            file_modifications[modified_file.filename] += 1

most_modified_files = file_modifications.most_common(5)

for filename, count in most_modified_files:
    print(f'{filename}: {count} modificações')

print("5. Quais os 5 arquivos “.java” ou / “.js” ou / “.py” mais modificados desde 2020?")
start_date = datetime(2020, 1, 1)

file_modifications = Counter()

for commit in Repository(url_junit, since=start_date).traverse_commits():
    for modified_file in commit.modified_files:
        if modified_file.filename.endswith(('.java', '.js', '.py')):
            file_modifications[modified_file.filename] += 1

most_modified_files = file_modifications.most_common(5)

for filename, count in most_modified_files:
    print(f'{filename}: {count} modificações')

print("6. Quais os 5 arquivos “.java” ou / “.js” ou / “.py” mais modificados em 2019?")
start_date = datetime(2019, 1, 1)
end_date = datetime(2019, 12, 31)

file_modifications = Counter()

for commit in Repository(url_junit, since=start_date, to=end_date).traverse_commits():
    for modified_file in commit.modified_files:
        if modified_file.filename.endswith(('.java', '.js', '.py')):
            file_modifications[modified_file.filename] += 1

most_modified_files = file_modifications.most_common(5)

for filename, count in most_modified_files:
    print(f'{filename}: {count} modificações')

print("7. Quais os 3 desenvolvedores mais ativos em termos de quantidade de commits?")
author_commits = Counter()

for commit in Repository(url_junit).traverse_commits():
    author_commits[commit.author.name] += 1

most_active_developers = author_commits.most_common(3)

for author, count in most_active_developers:
    print(f'{author}: {count} commits')

print(" 8. Quais 3 desenvolvedores mais ativos em 2019?")
start_date = datetime(2019, 1, 1)
end_date = datetime(2019, 12, 31)

author_commits = Counter()

for commit in Repository(url_junit, since=start_date, to=end_date).traverse_commits():
    author_commits[commit.author.name] += 1

most_active_developers = author_commits.most_common(3)

for author, count in most_active_developers:
    print(f'{author}: {count} commits')