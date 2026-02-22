# PoeCoder Tool Command Surface

This document lists model-callable tools from the runtime command catalog with compact argument/effect notes.

Protocol note:
- `@ask {"prompt":"...","key":"..."}` is a model-to-user clarification command (not a tool). CLI collects user input and continues automatically.

## File and code tools

- Help
  - args: `tool_name?/query?`
  - effect: return detailed usage for a specific tool (or list tools when omitted)
- ReadRaw
  - args: `file,line,end_line?`
  - effect: return raw file text by line window
- ReadStruct
  - args: `target,language,dependency_depth`
  - effect: return structural/AST summary and dependencies
- ReadRecursive (alias: ReadRecurisive)
  - args: `seed_files,boundary`
  - effect: recursively expand related implementation files
- Search
  - args: `pattern,file_pattern,boundary,root`
  - effect: regex matches with context snippets
- ListFile
  - args: `path?,pattern?,recursive?,include_dirs?,limit?`
  - effect: list files/dirs under a path
- ChangeWorkDir
  - args: `path`
  - effect: switch tool runtime working directory
- WriteRaw
  - args: `file,line,content,append?`
  - effect: insert/append text at target file location
- WriteReplace
  - args: `pattern,replacement,location,max_changes`
  - effect: regex replace in scoped files

## Web tools

- GetWebRaw
  - args: `url,timeout_s,max_chars,headers?,selector?,regex?,max_matches?`
  - effect: fetch web payload with optional selector/regex filtering
- GetWeb
  - args: `url,focus?,timeout_s,max_chars,selector?,regex?,max_matches?,download_if_large?,download_folder?`
  - effect: fetch/summarize page with optional filtering and large-page download fallback
- GetWebFile
  - args: `url,save_as?,folder,overwrite,timeout_s,max_bytes`
  - effect: download remote file

## Memory and wiki

- WriteMemory
  - args: `scope,content,tags?,priority,session_id?,project_id?`
  - effect: create memory entry
- ReadMemory
  - args: `scope?,query?,session_id?,project_id?,tags_any?,min_priority?,include_content?,max_content_chars?,limit`
  - effect: read filtered memory entries
- EditMemory
  - args: `entry_id?/query,operation,payload,scope?`
  - effect: mutate memory entries
- DelMemory
  - args: `entry_id?/query,scope?`
  - effect: delete memory entries
- WikiQuery
  - args: `project_id,query,topic?,include_content?,include_meta?,max_content_chars?,limit?`
  - effect: query project wiki with optional compact output
- WikiCompact
  - args: `project_id`
  - effect: compact wiki documents

## Command registry and temp artifacts

- TmpWrite
  - args: `name,content,ttl_seconds?`
  - effect: save temporary named content
- InstallCommand
  - args: `name,definition,runtime,args_schema,effect_schema,capabilities,source,signature?,session_id?`
  - effect: install/update reusable command
- EditCommand
  - args: `name,definition?/args_schema?/effect_schema?/capabilities?/signature?,session_id?`
  - effect: patch installed command
- DelCommand
  - args: `name,session_id?`
  - effect: delete installed command

## Models, review, and providers

- ListModels
  - args: `refresh?`
  - effect: list supported models
- ChangeModel
  - args: `session_id,model`
  - effect: change active model for a session
- Review
  - args: `session_id,prompt,context_keys?,model?,thinking_level?,thinking_budget?`
  - effect: run reviewer-role analysis
- GetBalance
  - args: none
  - effect: read Poe points balance
- SetBaseUri
  - args: `provider,base_uri`
  - effect: set model provider base URI (`poe` or `openai`)

## Subagents and background tasks

- StartSubAgent
  - args: `parent_session_id,model,perm,prompt,context_share,images?,system_message_modifier?`
  - effect: start subagent
- ReadSubAgent
  - args: `agent_id`
  - effect: read subagent state
- WaitSubAgent
  - args: `agent_id,timeout_s?`
  - effect: wait for subagent completion
- CancelSubAgent
  - args: `agent_id`
  - effect: cancel subagent
- StartBackgroundTurn
  - args: `session_id,user_prompt,system_message?,images?,context_keys?,metadata?`
  - effect: launch async turn task
- StartBackgroundSubAgent
  - args: `parent_session_id,model,perm,prompt,images?,context_share?,system_message_modifier?,wait_timeout_s?`
  - effect: launch async subagent task
- ListTasks
  - args: `limit?,state?,task_type?`
  - effect: list background tasks
- ReadTaskOutput
  - args: `task_id`
  - effect: read background task output
- CancelTask
  - args: `task_id`
  - effect: cancel background task

## Leader orchestration and shell

- StartLeaderRun
  - args: `session_id,goal,jobs?,planner_model?,worker_model?,max_parallel?,per_job_timeout_s?,context_keys?,verify_command?,verify_cwd?,verify_timeout_s?,verify_danger_level?`
  - effect: start scoped parallel leader run
- ReadLeaderRun
  - args: `run_id`
  - effect: read leader run status and result
- ListLeaderJobs
  - args: `run_id`
  - effect: list jobs for a leader run
- WaitLeaderRun
  - args: `run_id,timeout_s?`
  - effect: wait for leader run completion
- CancelLeaderRun
  - args: `run_id`
  - effect: cancel leader run
- RunShell
  - args: `session_id,command,danger_level,cwd?,timeout_s?`
  - effect: execute shell command through policy gate
- Exit
  - args: `reason?`
  - effect: signal CLI/session exit
