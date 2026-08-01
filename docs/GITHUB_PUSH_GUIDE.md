# GitHub Push Guide

## 1. 압축을 푼 폴더로 이동

```bat
cd /d "C:\Users\chefc\OneDrive\바탕 화면\YAI Challenge\CPR_STRING"
```

## 2. 선택 사항: 원본 330-case `cases.jsonl` 가져오기

```bat
scripts\IMPORT_LOCAL_RUN_WINDOWS.bat
```

대용량 generation cache와 API key는 `.gitignore`로 제외됩니다.

## 3. 새 저장소로 커밋하고 Push

```bat
git init
git branch -M main
git add .
git status
git commit -m "Unify STRING core and CPR applications"
git remote add origin https://github.com/curry2175/CPR_STRING.git
git push -u origin main
```

이미 `origin`이 있으면:

```bat
git remote set-url origin https://github.com/curry2175/CPR_STRING.git
git push -u origin main
```

원격 `CPR_STRING`을 이 통합본으로 완전히 교체하려는 경우에만:

```bat
git push -u origin main --force
```

## Push 전 비밀정보 확인

```bat
git status
git ls-files | findstr /i ".env generation_cache .venv"
```

`.env`, `.venv`, generation cache 또는 API key가 표시되면 push하지 마세요.
