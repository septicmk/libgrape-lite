/*
 * Copyright 2015 Delft University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package science.atlarge.graphalytics.libgrape.algorithms.wcc;

import java.util.List;

import org.apache.commons.configuration.Configuration;

import science.atlarge.graphalytics.libgrape.LibgrapeJob;

public class ConnectedComponentsJob extends LibgrapeJob {

	public ConnectedComponentsJob(Configuration config, String verticesPath, String edgesPath, boolean graphDirected, String jobId, String logPath) {
		super(config, verticesPath, edgesPath, graphDirected, jobId, logPath);
	}

	@Override
	protected void addJobArguments(List<String> args) {
		//args.add("--application");
		args.add("wcc");
	}
}
