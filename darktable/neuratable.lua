local dt = require "darktable"
local df = require "lib/dtutils.file"
require "darktable.debug"

-- Script data
local script_data = {}

script_data.metadata = {
  name = "process with Neuratable",
  purpose = "processing images with Neuratable",
  author = "Johannes May <johannes@fj-may.de>",
  help = ""
}

script_data.destroy = nil -- function to destroy the script
script_data.destroy_method = nil -- set to hide for libs since we can't destroy them commpletely yet, otherwise leave as nil
script_data.restart = nil -- how to restart the (lib) script after it's been hidden - i.e. make it visible again
script_data.show = nil -- only required for libs since the destroy_method only hides them

-- Your external tool
local tool = "neuratable_cli denoise"

-- Group leaders by directory
local function get_group_leaders_by_dir(selection)
  local grouped = {}

  for _, img in ipairs(selection) do
    local leader = img.group_leader or img
    local dir = leader.path

    grouped[dir] = grouped[dir] or {}
    local leader_id = leader.id
    if not grouped[dir][leader_id] then
      grouped[dir][leader_id] = leader
    end
  end

  return grouped
end

-- Run external tool per directory
local function denoise_with_neuratable()
  local selected = dt.gui.selection()

  if #selected == 0 then
    dt.print("No images selected.")
    return
  end

  local grouped_leaders = get_group_leaders_by_dir(selected)
  -- local leader_lookup = {}  -- Map new filenames to original leader image

  for dir, leaders in pairs(grouped_leaders) do
    local input_paths = {}
    for _, leader in pairs(leaders) do
      table.insert(input_paths, '"' .. leader.path .. "/" .. leader.filename .. '"')
    end

    local output_dir = dir .. "/denoised"
    df.mkdir(output_dir)

    local output_pattern = output_dir .. "/%NAME%_denoised.tif"
    local cmd = string.format('%s %s "%s"', tool, table.concat(input_paths, " "), output_pattern)
    dt.print_log("Running: " .. cmd)
    os.execute(cmd)

    -- Build a mapping of expected output files for regrouping
    for _, leader in pairs(leaders) do
      local name_base = df.get_basename(leader.filename)
      local new_filename = name_base .. "_denoised.tif"
      local new_filepath = output_dir .. "/" .. new_filename
      
      local new_img = dt.database.import(new_filepath)
      new_img:group_with(leader)    
      new_img:make_group_leader()
    end
  end

  dt.print("Denoising and regrouping complete.")
end

DenoiseWithNeuratable_btn_run = dt.new_widget("button"){
	label = "denoise with Neuratable",
	tooltip = "Denoise the selected images with Neuratable",
	clicked_callback = denoise_with_neuratable
	}
ProcessWithNeuratableBox =  dt.new_widget("box"){
		orientation = "vertical",
		DenoiseWithNeuratable_btn_run
	}
dt.register_lib(
	"Neuratable_Lib",	-- Module name
	"Process With Neuratable",	-- name
	true,	-- expandable
	false,	-- resettable
	{[dt.gui.views.lighttable] = {"DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 99}},	-- containers
	ProcessWithNeuratableBox,
	nil,
	nil
)

local function destroy()
  dt.print_log("Destroying Neuratable script")
  dt.gui.libs["Neuratable_Lib"].visible = false -- we haven't figured out how to destroy it yet, so we hide it for now
end

local function restart()
    dt.gui.libs["Neuratable_Lib"].visible = true -- the user wants to use it again, so we just make it visible and it shows up in the UI
end

script_data.destroy = destroy
script_data.restart = restart
script_data.destroy_method = "hide"

return script_data
