# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## General Instructions
- You must read all instructions in this file thoroughly to guarantee that absolutely no tasks are overlooked.
- When starting a new task in `Your Work`, please be sure to check the latest CLAUDE.md.

## Project Overview
HyperionResearcher is a workspace where Claude Code himself becomes an agent, exploring the Web, finding and summarizing security news.

## Your Main Function(Work)
Please execute the task defined in 'Your Work' "fully automatically" when the following respective commands are given:

- "SUMMARY"
    - `News collection using news_source.csv`

You don't need to pull, commit and push to git repo, it is my work.

## Your Work
The main tasks performed by Claude Code are...

- SUMMARY - News collection using news_source.csv
    - As per the instructions in my prompt, please access the designated security news websites listed within the news_source.csv file to gather the most recent security news, and save summary to markdown file.
        - Also, when searching for security news, please check the summary in the todays_news folder for the most recent week to ensure that there are no duplicates in the security news.
        - Finally, please translate the summary into Japanese. When doing so, please take the utmost care to ensure that the meaning of terminology specific to the security industry does not change. After translating the summary into Japanese, please delete the original English.
 
### Cautions
When summarizing, please **strictly** follow the points below.

#### Summarying
- Read the full text of the article, and include a 15-50 word title and a 150-300 word summary, along with the security news site title and news URLs (not the site URL). 
- Even if the news is from the same website, please separate it into different sections if it pertains to a different topic.

#### Summary Saving
- Please list `news_title`, `site_title`, `news_url`, `summary` in the todays_news folder in <date of the day>.md (for example, 250614.md if you work in 2025/06/14). You must not use any naming convention other than this one, that is, other than the "6-digit number.md" format.
- It is possible that the same task you are performing was already done at another time today, and today's file already exists. In that case, please append your additions to that file. If the event that the file is currently in use by another process, implement a **waiting** mechanism until access is granted to guarantee a successful write operation.