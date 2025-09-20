/**
 * FIFO Queues for Judgment Processing
 * ==================================
 *
 * This module implements queue data structures for judgment import and processing
 * workflows in Find Case Law (FCL). Provides first-in-first-out (FIFO) processing
 * for document ingestion, analysis, and publishing pipelines.
 *
 * Key FCL Use Cases:
 * - Judgment import/processing queue management
 * - Background task scheduling for document analysis
 * - Bulk document operations with proper ordering
 * - Async processing coordination between services
 * - Rate-limited operations for external API calls
 */

// Priority levels for judgment processing
const Priority = {
    CRITICAL: 1,    // System failures, urgent fixes
    HIGH: 2,        // Supreme Court, House of Lords judgments
    NORMAL: 3,      // Court of Appeal, High Court judgments
    LOW: 4          // Tribunal decisions, bulk imports
};

// Status of queued tasks
const TaskStatus = {
    PENDING: 'pending',
    PROCESSING: 'processing',
    COMPLETED: 'completed',
    FAILED: 'failed',
    RETRYING: 'retrying'
};

/**
 * Represents a judgment processing task in the queue
 */
class JudgmentTask {
    constructor(taskId, judgmentId, operation, priority, metadata = {}) {
        this.taskId = taskId;
        this.judgmentId = judgmentId;
        this.operation = operation; // 'import', 'reprocess', 'enrich', 'publish'
        this.priority = priority;
        this.createdAt = new Date();
        this.attempts = 0;
        this.maxAttempts = 3;
        this.status = TaskStatus.PENDING;
        this.errorMessage = null;
        this.metadata = metadata;
    }

    toString() {
        const priorityName = Object.keys(Priority).find(key => Priority[key] === this.priority);
        return `Task ${this.taskId}: ${this.operation} ${this.judgmentId} (${priorityName})`;
    }
}

/**
 * Abstract base class for queue implementations
 */
class Queue {
    enqueue(item) {
        throw new Error('Abstract method must be implemented');
    }

    dequeue() {
        throw new Error('Abstract method must be implemented');
    }

    peek() {
        throw new Error('Abstract method must be implemented');
    }

    size() {
        throw new Error('Abstract method must be implemented');
    }

    isEmpty() {
        throw new Error('Abstract method must be implemented');
    }
}

/**
 * Simple FIFO queue implementation using array for efficient operations.
 * Suitable for basic judgment processing workflows.
 */
class SimpleQueue extends Queue {
    constructor() {
        super();
        this._queue = [];
    }

    /**
     * Add item to the end of queue (FIFO)
     */
    enqueue(item) {
        this._queue.push(item);
    }

    /**
     * Remove and return item from front of queue
     */
    dequeue() {
        if (this._queue.length === 0) {
            return null;
        }
        return this._queue.shift();
    }

    /**
     * View next item without removing it
     */
    peek() {
        if (this._queue.length === 0) {
            return null;
        }
        return this._queue[0];
    }

    /**
     * Get current queue size
     */
    size() {
        return this._queue.length;
    }

    /**
     * Check if queue is empty
     */
    isEmpty() {
        return this._queue.length === 0;
    }

    /**
     * Clear all items from queue
     */
    clear() {
        this._queue = [];
    }

    /**
     * Return queue contents as array (for inspection)
     */
    toArray() {
        return [...this._queue];
    }
}

/**
 * Priority queue for judgment processing tasks.
 * Higher priority items are processed first.
 */
class PriorityQueue extends Queue {
    constructor() {
        super();
        this._queues = new Map();

        // Initialize queues for each priority level
        Object.values(Priority).forEach(priority => {
            this._queues.set(priority, new SimpleQueue());
        });
    }

    /**
     * Add task to appropriate priority queue
     */
    enqueue(task) {
        const queue = this._queues.get(task.priority);
        if (queue) {
            queue.enqueue(task);
        } else {
            throw new Error(`Invalid priority: ${task.priority}`);
        }
    }

    /**
     * Remove and return highest priority task
     */
    dequeue() {
        // Check queues in priority order (lowest number = highest priority)
        const priorities = Object.values(Priority).sort((a, b) => a - b);

        for (const priority of priorities) {
            const queue = this._queues.get(priority);
            if (!queue.isEmpty()) {
                return queue.dequeue();
            }
        }
        return null;
    }

    /**
     * View next highest priority task without removing
     */
    peek() {
        const priorities = Object.values(Priority).sort((a, b) => a - b);

        for (const priority of priorities) {
            const queue = this._queues.get(priority);
            if (!queue.isEmpty()) {
                return queue.peek();
            }
        }
        return null;
    }

    /**
     * Get total number of tasks across all priorities
     */
    size() {
        let total = 0;
        for (const queue of this._queues.values()) {
            total += queue.size();
        }
        return total;
    }

    /**
     * Check if all priority queues are empty
     */
    isEmpty() {
        for (const queue of this._queues.values()) {
            if (!queue.isEmpty()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Get count of tasks by priority level
     */
    getPriorityCounts() {
        const counts = {};
        const priorityNames = Object.keys(Priority);

        for (const [priority, queue] of this._queues.entries()) {
            const priorityName = priorityNames.find(name => Priority[name] === priority);
            counts[priorityName] = queue.size();
        }
        return counts;
    }
}

/**
 * High-level judgment processing system using queues.
 * Coordinates multiple processing stages with proper error handling.
 */
class JudgmentProcessor {
    constructor(maxWorkers = 3) {
        this.processingQueue = new PriorityQueue();
        this.completedQueue = new SimpleQueue();
        this.failedQueue = new SimpleQueue();
        this.maxWorkers = maxWorkers;
        this.workersRunning = false;
        this.workers = [];
        this._stats = {
            processed: 0,
            failed: 0,
            retries: 0,
            startTime: null
        };

        // Mock processing functions for different operations
        this._processors = {
            'import': this._processImport.bind(this),
            'reprocess': this._processReprocess.bind(this),
            'enrich': this._processEnrichment.bind(this),
            'publish': this._processPublish.bind(this)
        };
    }

    /**
     * Add a judgment processing task to the queue
     */
    addJudgmentTask(judgmentId, operation, priority = Priority.NORMAL, metadata = {}) {
        const taskId = `${operation}_${judgmentId}_${Date.now()}`;

        const task = new JudgmentTask(
            taskId,
            judgmentId,
            operation,
            priority,
            metadata
        );

        this.processingQueue.enqueue(task);
        return taskId;
    }

    /**
     * Start worker processes to handle queue
     */
    async startProcessing() {
        if (this.workersRunning) {
            return;
        }

        this.workersRunning = true;
        this._stats.startTime = new Date();

        for (let i = 0; i < this.maxWorkers; i++) {
            const worker = this._createWorker(i);
            this.workers.push(worker);
        }

        console.log(`Started ${this.maxWorkers} workers for judgment processing`);
    }

    /**
     * Stop all worker processes
     */
    stopProcessing() {
        this.workersRunning = false;
        this.workers = [];
        console.log('Stopped all processing workers');
    }

    /**
     * Get comprehensive queue status
     */
    getQueueStatus() {
        const priorityCounts = this.processingQueue.getPriorityCounts();

        return {
            pendingTasks: this.processingQueue.size(),
            priorityBreakdown: priorityCounts,
            completedTasks: this.completedQueue.size(),
            failedTasks: this.failedQueue.size(),
            workersActive: this.workersRunning,
            workerCount: this.workers.length,
            statistics: { ...this._stats }
        };
    }

    /**
     * Get recently completed tasks
     */
    getRecentCompleted(limit = 10) {
        const completed = this.completedQueue.toArray();
        return completed.slice(-limit);
    }

    /**
     * Get all failed tasks for review
     */
    getFailedTasks() {
        return this.failedQueue.toArray();
    }

    /**
     * Retry a specific failed task
     */
    retryFailedTask(taskId) {
        const failedTasks = this.failedQueue.toArray();
        const taskIndex = failedTasks.findIndex(task => task.taskId === taskId);

        if (taskIndex === -1) {
            return false;
        }

        const task = failedTasks[taskIndex];

        // Reset task for retry
        task.status = TaskStatus.PENDING;
        task.attempts = 0;
        task.errorMessage = null;

        // Remove from failed queue and add back to processing
        this.failedQueue._queue.splice(taskIndex, 1);
        this.processingQueue.enqueue(task);
        return true;
    }

    /**
     * Create a worker process
     */
    _createWorker(workerId) {
        const worker = setInterval(async () => {
            if (!this.workersRunning) {
                clearInterval(worker);
                return;
            }

            try {
                const task = this.processingQueue.dequeue();
                if (task) {
                    await this._processTask(task, workerId);
                }
            } catch (error) {
                console.error(`Worker ${workerId} error:`, error);
            }
        }, 100); // Check for tasks every 100ms

        return worker;
    }

    /**
     * Process a single judgment task
     */
    async _processTask(task, workerId) {
        task.status = TaskStatus.PROCESSING;
        task.attempts++;

        console.log(`Worker ${workerId} processing: ${task.toString()}`);

        try {
            const processor = this._processors[task.operation];
            if (!processor) {
                throw new Error(`Unknown operation: ${task.operation}`);
            }

            const success = await processor(task);

            if (success) {
                task.status = TaskStatus.COMPLETED;
                this.completedQueue.enqueue(task);
                this._stats.processed++;
                console.log(` Completed: ${task.taskId}`);
            } else {
                this._handleTaskFailure(task);
            }
        } catch (error) {
            task.errorMessage = error.message;
            this._handleTaskFailure(task);
        }
    }

    /**
     * Handle task processing failure with retry logic
     */
    _handleTaskFailure(task) {
        if (task.attempts < task.maxAttempts) {
            // Retry the task
            task.status = TaskStatus.RETRYING;
            this.processingQueue.enqueue(task);
            this._stats.retries++;
            console.log(`  Retrying: ${task.taskId} (attempt ${task.attempts}/${task.maxAttempts})`);
        } else {
            // Task failed permanently
            task.status = TaskStatus.FAILED;
            this.failedQueue.enqueue(task);
            this._stats.failed++;
            console.log(` Failed: ${task.taskId} - ${task.errorMessage}`);
        }
    }

    /**
     * Simulate judgment import processing
     */
    async _processImport(task) {
        const processingTime = {
            [Priority.CRITICAL]: 100,
            [Priority.HIGH]: 200,
            [Priority.NORMAL]: 500,
            [Priority.LOW]: 1000
        };

        await this._sleep(processingTime[task.priority]);

        // Simulate occasional failures for demonstration
        if (task.judgmentId.toLowerCase().includes('fail')) {
            return false;
        }

        return true;
    }

    /**
     * Simulate judgment reprocessing
     */
    async _processReprocess(task) {
        await this._sleep(300);
        return true;
    }

    /**
     * Simulate judgment enrichment (NLP, citations, etc.)
     */
    async _processEnrichment(task) {
        await this._sleep(800);
        return true;
    }

    /**
     * Simulate judgment publishing
     */
    async _processPublish(task) {
        await this._sleep(200);
        return true;
    }

    /**
     * Utility method for simulating processing delays
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Fixed-size circular buffer implementation.
 * Useful for maintaining recent processing history or logs.
 */
class CircularBuffer extends Queue {
    constructor(capacity) {
        super();
        if (capacity <= 0) {
            throw new Error('Capacity must be positive');
        }

        this.capacity = capacity;
        this._buffer = new Array(capacity);
        this._head = 0;  // Points to next write position
        this._tail = 0;  // Points to next read position
        this._size = 0;
    }

    /**
     * Add item to buffer, overwriting oldest if full
     */
    enqueue(item) {
        this._buffer[this._head] = item;
        this._head = (this._head + 1) % this.capacity;

        if (this._size < this.capacity) {
            this._size++;
        } else {
            // Buffer is full, advance tail
            this._tail = (this._tail + 1) % this.capacity;
        }
    }

    /**
     * Remove and return oldest item
     */
    dequeue() {
        if (this._size === 0) {
            return null;
        }

        const item = this._buffer[this._tail];
        this._buffer[this._tail] = null;
        this._tail = (this._tail + 1) % this.capacity;
        this._size--;
        return item;
    }

    /**
     * View oldest item without removing
     */
    peek() {
        if (this._size === 0) {
            return null;
        }
        return this._buffer[this._tail];
    }

    /**
     * Get current number of items
     */
    size() {
        return this._size;
    }

    /**
     * Check if buffer is empty
     */
    isEmpty() {
        return this._size === 0;
    }

    /**
     * Check if buffer is at capacity
     */
    isFull() {
        return this._size === this.capacity;
    }

    /**
     * Return buffer contents as array (oldest to newest)
     */
    toArray() {
        if (this._size === 0) {
            return [];
        }

        const result = [];
        let current = this._tail;
        for (let i = 0; i < this._size; i++) {
            result.push(this._buffer[current]);
            current = (current + 1) % this.capacity;
        }
        return result;
    }
}

/**
 * Demonstrate queue implementations with UK legal data
 */
async function demonstrateJudgmentQueues() {
    console.log('=== Judgment Processing Queues Demo ===\n');

    // Sample UK judgments for processing
    const ukJudgments = [
        {
            id: 'uksc_2023_15',
            caseName: 'R (Miller) v Prime Minister',
            citation: '[2023] UKSC 15',
            court: 'UKSC',
            priority: Priority.HIGH
        },
        {
            id: 'ewca_2023_892',
            caseName: 'Smith v Secretary of State',
            citation: '[2023] EWCA Civ 892',
            court: 'EWCA',
            priority: Priority.NORMAL
        },
        {
            id: 'ewhc_2023_1456',
            caseName: 'Jones v Local Authority',
            citation: '[2023] EWHC 1456 (Admin)',
            court: 'EWHC',
            priority: Priority.NORMAL
        },
        {
            id: 'ukftt_2023_fail',  // This will simulate a failure
            caseName: 'Brown v HMRC',
            citation: '[2023] UKFTT 234 (TC)',
            court: 'UKFTT',
            priority: Priority.LOW
        }
    ];

    // 1. Simple Queue Demo
    console.log('1. SIMPLE FIFO QUEUE:');
    const simpleQueue = new SimpleQueue();

    ukJudgments.forEach(judgment => {
        simpleQueue.enqueue(judgment.caseName);
        console.log(`   Enqueued: ${judgment.caseName}`);
    });

    console.log(`\n   Queue size: ${simpleQueue.size()}`);
    console.log(`   Next item: ${simpleQueue.peek()}`);

    console.log(`\n   Processing queue:`);
    while (!simpleQueue.isEmpty()) {
        const caseName = simpleQueue.dequeue();
        console.log(`   Processing: ${caseName}`);
    }

    // 2. Priority Queue Demo
    console.log(`\n2. PRIORITY QUEUE:`);
    const processor = new JudgmentProcessor(2);

    // Add tasks with different priorities
    ukJudgments.forEach(judgment => {
        const taskId = processor.addJudgmentTask(
            judgment.id,
            'import',
            judgment.priority,
            { court: judgment.court, citation: judgment.citation }
        );
        const priorityName = Object.keys(Priority).find(key => Priority[key] === judgment.priority);
        console.log(`   Added task: ${taskId} (${priorityName})`);
    });

    // Add some enrichment tasks
    processor.addJudgmentTask('uksc_2023_15', 'enrich', Priority.HIGH);
    processor.addJudgmentTask('ewca_2023_892', 'publish', Priority.NORMAL);

    // Show queue status
    let status = processor.getQueueStatus();
    console.log(`\n   Queue status:`);
    console.log(`     Pending tasks: ${status.pendingTasks}`);
    console.log(`     Priority breakdown: ${JSON.stringify(status.priorityBreakdown)}`);

    // 3. Process Tasks
    console.log(`\n3. PROCESSING TASKS:`);
    await processor.startProcessing();

    // Wait for processing to complete
    await new Promise(resolve => setTimeout(resolve, 3000));

    const finalStatus = processor.getQueueStatus();
    console.log(`\n   Final status:`);
    console.log(`     Completed: ${finalStatus.completedTasks}`);
    console.log(`     Failed: ${finalStatus.failedTasks}`);
    console.log(`     Statistics: ${JSON.stringify(finalStatus.statistics)}`);

    // Show completed tasks
    const completed = processor.getRecentCompleted(5);
    if (completed.length > 0) {
        console.log(`\n   Recent completions:`);
        completed.forEach(task => {
            console.log(`      ${task.judgmentId} (${task.operation})`);
        });
    }

    // Show failed tasks
    const failed = processor.getFailedTasks();
    if (failed.length > 0) {
        console.log(`\n   Failed tasks:`);
        failed.forEach(task => {
            console.log(`      ${task.judgmentId}: ${task.errorMessage}`);
        });
    }

    processor.stopProcessing();

    // 4. Circular Buffer Demo
    console.log(`\n4. CIRCULAR BUFFER (Recent History):`);
    const historyBuffer = new CircularBuffer(3);

    const events = [
        'Import started for UKSC 2023/15',
        'Enrichment completed for EWCA 2023/892',
        'Publishing failed for EWHC 2023/1456',
        'Retry successful for EWHC 2023/1456',
        'New batch import initiated'
    ];

    events.forEach(event => {
        historyBuffer.enqueue(event);
        console.log(`   Event: ${event}`);
        if (historyBuffer.isFull()) {
            console.log(`   Buffer full - oldest events will be overwritten`);
        }
    });

    console.log(`\n   Recent history (last ${historyBuffer.size()} events):`);
    const history = historyBuffer.toArray();
    history.forEach((event, index) => {
        console.log(`     ${index + 1}. ${event}`);
    });

    return {
        simpleQueue,
        processor,
        historyBuffer,
        finalStats: finalStatus
    };
}

// Export classes and functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        Priority,
        TaskStatus,
        JudgmentTask,
        Queue,
        SimpleQueue,
        PriorityQueue,
        JudgmentProcessor,
        CircularBuffer,
        demonstrateJudgmentQueues
    };
}

// Run demonstration if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
    demonstrateJudgmentQueues().catch(console.error);
}