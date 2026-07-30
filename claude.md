# Claude Coding Best Practices Guide

This guide defines the coding standards and practices that Claude will follow when you ask to build something. These principles ensure clean, maintainable, robust, and performant code.

---

## 01 Coding Style & Consistency

Follow strict coding style standards for clarity and maintainability.

**Principles:**
- Adhere to language-specific conventions and linter rules
- Maintain consistent naming conventions (camelCase, snake_case, PascalCase as appropriate)
- Use consistent indentation (spaces or tabs as per project standard)
- Keep lines reasonably short (aim for 80-120 characters)
- Use meaningful variable and function names that are self-documenting

**When Building:**
- If a `.eslintrc`, `.prettier`, or similar config file exists, follow it strictly
- If no style guide is specified, use language defaults (PEP 8 for Python, Google style for Java, etc.)
- Consistency trumps personal preference

**Example:**
```javascript
// ❌ Inconsistent style
function getUserData(id){
const u = db.query("SELECT * FROM users WHERE id=" + id)
return u
}

// ✅ Consistent style
function getUserData(userId) {
  const user = db.query('SELECT * FROM users WHERE id = ?', [userId]);
  return user;
}
```

---

## 02 Write Small, Focused Changes

Break complex implementations into smaller, manageable chunks, each representing a single coherent change.

**Why Small Changes Matter:**
- **Easier Review:** Reviewers can understand each change clearly
- **Faster Debugging:** Isolate issues to specific, focused changes
- **Reduced Conflicts:** Minimize merge conflicts by touching fewer files
- **Easier Reverting:** Revert specific functionality without affecting other code
- **Better Testing:** Validate each change independently

**How to Structure:**
- Each change should address ONE concern
- Separate concerns: setup → implementation → tests → documentation
- If a feature requires multiple changes, structure them logically
- Commit frequently with clear commit messages

**Example Structure for Building an Email Validator:**
1. **Change 1:** Add basic email validation function with regex
2. **Change 2:** Add comprehensive RFC 5322 validation logic
3. **Change 3:** Add unit tests for happy path and edge cases
4. **Change 4:** Add error handling and validation messages
5. **Change 5:** Add integration tests and documentation

---

## 03 Write Clean, Robust, and Modular Code

Code should be simple, easy to understand, and hard to break.

**Core Principles:**

### Prefer Strong Typing
```typescript
// ❌ Weak typing
function process(data) {
  return data.map(x => x * 2);
}

// ✅ Strong typing
function processNumbers(numbers: number[]): number[] {
  return numbers.map(num => num * 2);
}
```

### Keep Functions Small and Focused
```javascript
// ❌ Large function doing multiple things
function getUserAndValidate(id) {
  const user = db.query(`SELECT * FROM users WHERE id = ${id}`);
  if (!user.email) throw new Error('Missing email');
  if (!user.email.includes('@')) throw new Error('Invalid email');
  const posts = db.query(`SELECT * FROM posts WHERE user_id = ${id}`);
  return { user, postCount: posts.length };
}

// ✅ Small, focused functions
function getUser(userId: number): User {
  return db.query('SELECT * FROM users WHERE id = ?', [userId]);
}

function validateUserEmail(user: User): void {
  if (!user.email) throw new Error('Email is required');
  if (!isValidEmail(user.email)) throw new Error('Invalid email format');
}

function getUserWithPostCount(userId: number): UserWithStats {
  const user = getUser(userId);
  validateUserEmail(user);
  const posts = db.query('SELECT COUNT(*) FROM posts WHERE user_id = ?', [userId]);
  return { ...user, postCount: posts[0].count };
}
```

### Extract Shared Functionality (DRY Principle)
```javascript
// ❌ Repeated validation logic
function createUser(email) {
  if (!email || !email.includes('@')) throw new Error('Invalid email');
  // ... create user
}

function updateUser(id, email) {
  if (!email || !email.includes('@')) throw new Error('Invalid email');
  // ... update user
}

// ✅ Shared utility function
function validateEmail(email: string): void {
  if (!email) throw new Error('Email is required');
  if (!email.includes('@')) throw new Error('Invalid email format');
}

function createUser(email) {
  validateEmail(email);
  // ... create user
}

function updateUser(id, email) {
  validateEmail(email);
  // ... update user
}
```

### Defensive Programming
```javascript
// ✅ Guard against unexpected inputs
function calculateAverage(numbers: number[]): number {
  if (!Array.isArray(numbers) || numbers.length === 0) {
    throw new Error('Input must be a non-empty array');
  }
  
  if (!numbers.every(n => typeof n === 'number')) {
    throw new Error('All elements must be numbers');
  }
  
  return numbers.reduce((a, b) => a + b, 0) / numbers.length;
}
```

### Simplify Class Hierarchies
```javascript
// ❌ Overly complex inheritance
class Shape { }
class Polygon extends Shape { }
class Quadrilateral extends Polygon { }
class Rectangle extends Quadrilateral { }

// ✅ Simpler, composition-based approach
interface Shape {
  area(): number;
  perimeter(): number;
}

class Rectangle implements Shape {
  width: number;
  height: number;
  
  area() { return this.width * this.height; }
  perimeter() { return 2 * (this.width + this.height); }
}
```

### Minimize Parameters
```javascript
// ❌ Too many parameters
function createUser(firstName, lastName, email, phone, address, city, state, zip, country) {
  // ...
}

// ✅ Use objects to group related parameters
interface UserInput {
  firstName: string;
  lastName: string;
  email: string;
  phone?: string;
  address?: {
    street: string;
    city: string;
    state: string;
    zip: string;
    country: string;
  };
}

function createUser(input: UserInput) {
  // ...
}
```

---

## 04 Input Validation

Always validate inputs early and thoroughly.

**Validation Strategy:**

### Validate Early
```javascript
// ❌ Validate late (at usage point)
function processOrder(orderId) {
  const order = db.query('SELECT * FROM orders WHERE id = ?', [orderId]);
  // orderId might be invalid, but we don't know until here
  return order;
}

// ✅ Validate early (at function entry)
function processOrder(orderId: number) {
  if (!Number.isInteger(orderId) || orderId <= 0) {
    throw new Error('Order ID must be a positive integer');
  }
  
  const order = db.query('SELECT * FROM orders WHERE id = ?', [orderId]);
  return order;
}
```

### Use Strong Validation
```typescript
// ✅ Comprehensive validation rules
interface EmailValidationRules {
  minLength: number;
  maxLength: number;
  format: RegExp;
  allowedDomains?: string[];
}

function validateEmail(email: string, rules: EmailValidationRules): void {
  // Check length
  if (email.length < rules.minLength) {
    throw new Error(`Email must be at least ${rules.minLength} characters`);
  }
  
  if (email.length > rules.maxLength) {
    throw new Error(`Email must be at most ${rules.maxLength} characters`);
  }
  
  // Check format
  if (!rules.format.test(email)) {
    throw new Error('Invalid email format');
  }
  
  // Check allowed domains if specified
  if (rules.allowedDomains) {
    const domain = email.split('@')[1];
    if (!rules.allowedDomains.includes(domain)) {
      throw new Error(`Domain not in allowed list: ${rules.allowedDomains.join(', ')}`);
    }
  }
}
```

### Document Validation Rules
```javascript
/**
 * Validates a user email address
 * 
 * Rules:
 * - Must be a non-empty string
 * - Must follow RFC 5322 format
 * - Length: 5-254 characters (RFC 5321)
 * - Domains must be from approved list
 * 
 * @param {string} email - The email to validate
 * @throws {Error} If email is invalid
 */
function validateEmail(email: string): void {
  // Implementation
}
```

---

## 05 Assert Invariants

Use assertions to document and catch violations of expected conditions.

**Why Assert Invariants:**
- Catch bugs during development and testing
- Document assumptions clearly
- Prevent invalid state in production
- Improve debugging by failing fast

**Best Practices:**
```typescript
// ✅ Assert function preconditions
function withdrawMoney(account: Account, amount: number): void {
  // Preconditions
  console.assert(account !== null, 'Account must not be null');
  console.assert(amount > 0, 'Amount must be positive');
  console.assert(amount <= account.balance, 'Insufficient funds');
  
  account.balance -= amount;
  
  // Postconditions
  console.assert(account.balance >= 0, 'Balance cannot be negative');
}

// ✅ Assert loop invariants
function bubbleSort(arr: number[]): number[] {
  const n = arr.length;
  
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n - i - 1; j++) {
      // Invariant: elements after index n-i are sorted
      console.assert(isSorted(arr.slice(n - i)), 'Sorted portion violated');
      
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
      }
    }
  }
  
  return arr;
}
```

---

## 06 Code Comments: Explain the "Why"

Comments should explain reasoning and intent, not just restate what the code does.

**Comment Best Practices:**

### Explain Complex Logic
```javascript
// ❌ Comments restate obvious code
function calculateDiscount(price, quantity) {
  // Multiply price by quantity
  let total = price * quantity;
  
  // If quantity is greater than 10, discount is 10%
  if (quantity > 10) {
    total = total * 0.9;
  }
  
  return total;
}

// ✅ Comments explain the "why"
function calculateDiscount(price: number, quantity: number): number {
  let total = price * quantity;
  
  // Apply 10% bulk discount for orders over 10 units to encourage
  // larger purchases and improve inventory turnover for seasonal items
  if (quantity > 10) {
    total = total * 0.9;
  }
  
  return total;
}
```

### Document Workarounds and Assumptions
```javascript
// ✅ Document non-obvious decisions
function getUserProfile(userId: number): UserProfile {
  // NOTE: We cache the user profile for 5 minutes because the user_profile
  // view is expensive to compute and user data doesn't change frequently.
  // If this becomes a problem, consider using a message queue instead.
  // See: https://jira.company.com/browse/PERF-1234
  return cache.getOrSet(`user:${userId}`, () => {
    return db.query('SELECT * FROM user_profile WHERE id = ?', [userId]);
  }, 300000); // 5 minutes
}
```

### Comment Unusual Approaches
```javascript
// ✅ Explain why a less obvious approach was chosen
function findDuplicate(arr: number[]): number {
  // We use the Floyd's cycle detection algorithm (tortoise and hare)
  // instead of a Set because it has O(1) space complexity while finding
  // duplicates in a range [1, n]. See: https://en.wikipedia.org/wiki/Cycle_detection
  
  let slow = arr[0];
  let fast = arr[0];
  
  do {
    slow = arr[slow];
    fast = arr[arr[fast]];
  } while (slow !== fast);
  
  // Find the entrance to the cycle
  slow = arr[0];
  while (slow !== fast) {
    slow = arr[slow];
    fast = arr[fast];
  }
  
  return slow;
}
```

### Use Full Sentences
```javascript
// ✅ Comments use proper grammar and punctuation
// Fetch the user profile from the cache to avoid repeated database queries
// for the same user within a 5-minute window, which would improve response times.
const profile = cache.get(`user:${id}`);

// ❌ Incomplete comments
// get profile from cache
const profile = cache.get(`user:${id}`);
```

---

## 07 Effective Error Handling

Handle errors thoughtfully based on whether they're exceptional or expected.

**Ask These Questions:**
1. Is this an exceptional failure that should alert developers?
2. Is this a non-critical failure the code can handle gracefully?

**Pattern 1: Throw Exceptions for Exceptional Cases**
```typescript
// ✅ Throw for truly exceptional conditions
function connectToDatabase(connectionString: string): Connection {
  if (!connectionString) {
    throw new Error('Connection string is required');
  }
  
  try {
    return db.connect(connectionString);
  } catch (error) {
    // Database being down is exceptional; propagate the error
    throw new Error(`Failed to connect to database: ${error.message}`);
  }
}
```

**Pattern 2: Handle and Return Default for Expected Cases**
```typescript
// ✅ Try-catch for expected, non-critical failures
function fetchUserPreferences(userId: number): UserPreferences {
  try {
    return db.query('SELECT * FROM user_preferences WHERE id = ?', [userId]);
  } catch (error) {
    // User might not have preferences yet; return defaults
    console.warn(`Could not fetch preferences for user ${userId}: ${error.message}`);
    return getDefaultPreferences();
  }
}
```

**Pattern 3: Validate and Return Error Message**
```typescript
// ✅ Validate user input and return friendly error
function updateUserEmail(userId: number, newEmail: string): { success: boolean; error?: string } {
  // Validation errors are expected; return friendly message
  if (!newEmail || !newEmail.includes('@')) {
    return { success: false, error: 'Please provide a valid email address' };
  }
  
  try {
    db.query('UPDATE users SET email = ? WHERE id = ?', [newEmail, userId]);
    return { success: true };
  } catch (error) {
    // Database errors are exceptional; log and return generic message
    console.error(`Failed to update email for user ${userId}:`, error);
    return { success: false, error: 'Unable to update email. Please try again later.' };
  }
}
```

---

## 08 Write Comprehensive Tests

Always include tests in the same commit as code changes.

**Testing Strategy:**

### Unit Tests for Core Logic
```typescript
// ✅ Comprehensive unit tests
describe('validateEmail', () => {
  // Happy path
  it('should accept valid email addresses', () => {
    expect(validateEmail('user@example.com')).toBe(true);
    expect(validateEmail('first.last+tag@example.co.uk')).toBe(true);
  });
  
  // Edge cases
  it('should reject emails without @', () => {
    expect(validateEmail('userexample.com')).toBe(false);
  });
  
  it('should reject emails with multiple @', () => {
    expect(validateEmail('user@@example.com')).toBe(false);
  });
  
  // Boundary conditions
  it('should reject extremely long emails', () => {
    const longEmail = 'a'.repeat(300) + '@example.com';
    expect(validateEmail(longEmail)).toBe(false);
  });
  
  // Security concerns
  it('should reject emails with potential injection attempts', () => {
    expect(validateEmail('user@example.com\n<script>alert("xss")</script>')).toBe(false);
  });
  
  // Error handling
  it('should throw if input is not a string', () => {
    expect(() => validateEmail(null)).toThrow();
    expect(() => validateEmail(123)).toThrow();
  });
});
```

### Integration Tests
```typescript
// ✅ Test interactions between components
describe('User Registration', () => {
  it('should create user and send welcome email', async () => {
    const user = await registerUser({
      email: 'newuser@example.com',
      password: 'SecurePass123'
    });
    
    expect(user.id).toBeDefined();
    expect(user.email).toBe('newuser@example.com');
    expect(emailService.sendWelcomeEmail).toHaveBeenCalledWith('newuser@example.com');
  });
  
  it('should reject duplicate email addresses', async () => {
    await registerUser({ email: 'user@example.com', password: 'Pass123' });
    
    await expect(
      registerUser({ email: 'user@example.com', password: 'Pass456' })
    ).rejects.toThrow('Email already in use');
  });
});
```

### Test Coverage Requirements
- Aim for 80%+ code coverage
- Prioritize critical paths first (business logic, security, data mutations)
- Include edge cases and error scenarios
- Test privacy and security implications

---

## 09 Carefully Design APIs

Components should be deep and encapsulate complexity; APIs should be narrow and minimize dependencies.

**API Design Principles:**

### Hide Implementation Details
```typescript
// ❌ Exposes internal implementation
class UserManager {
  users: Map<number, User> = new Map();
  
  getUser(id: number): User {
    return this.users.get(id); // Exposes internal Map
  }
}

// ✅ Hides implementation, provides clean interface
class UserManager {
  private users: Map<number, User> = new Map();
  
  getUser(id: number): User | null {
    const user = this.users.get(id);
    if (!user) {
      throw new UserNotFoundError(id);
    }
    return user;
  }
}
```

### Minimize Dependencies
```typescript
// ❌ Too many parameters coupling components
function createUserProfile(
  user: User,
  preferences: UserPreferences,
  settings: AppSettings,
  cache: CacheService,
  logger: Logger,
  emailService: EmailService,
  analyticsService: AnalyticsService
) {
  // ...
}

// ✅ Pass only necessary context
interface UserProfileContext {
  preferences: UserPreferences;
  locale: string;
}

function createUserProfile(user: User, context: UserProfileContext): UserProfile {
  // ...
}
```

### Provide Consistent Interfaces
```typescript
// ✅ Consistent API across similar classes
interface Repository<T> {
  create(item: T): Promise<T>;
  read(id: string): Promise<T>;
  update(id: string, item: T): Promise<T>;
  delete(id: string): Promise<void>;
}

class UserRepository implements Repository<User> {
  // Implementation
}

class PostRepository implements Repository<Post> {
  // Implementation
}
```

---

## 10 Performance Considerations

Keep performance in mind throughout implementation.

**Performance Strategies:**

### Use Efficient Data Structures
```typescript
// ❌ O(n) lookup time
function hasRole(roles: string[], targetRole: string): boolean {
  return roles.includes(targetRole);
}

// ✅ O(1) lookup time
function hasRole(roles: Set<string>, targetRole: string): boolean {
  return roles.has(targetRole);
}
```

### Apply Filters Early
```typescript
// ❌ Process all data then filter
function getActiveUserEmails(): string[] {
  const allUsers = db.query('SELECT * FROM users');
  const activeUsers = allUsers.filter(u => u.isActive);
  return activeUsers.map(u => u.email);
}

// ✅ Filter at source
function getActiveUserEmails(): string[] {
  return db.query('SELECT email FROM users WHERE isActive = true');
}
```

### Use Caching for Expensive Operations
```typescript
// ✅ Memoize expensive function calls within request
function expensiveComputation(input: string): Result {
  const memoKey = `computation:${input}`;
  
  if (requestCache.has(memoKey)) {
    return requestCache.get(memoKey);
  }
  
  const result = slowFunction(input);
  requestCache.set(memoKey, result);
  return result;
}
```

### Use Concurrency Where Possible
```typescript
// ❌ Sequential operations
async function getUserWithStats(userId: number): Promise<UserStats> {
  const user = await getUser(userId);
  const posts = await getPostCount(userId);
  const followers = await getFollowerCount(userId);
  
  return { user, posts, followers };
}

// ✅ Parallel operations
async function getUserWithStats(userId: number): Promise<UserStats> {
  const [user, posts, followers] = await Promise.all([
    getUser(userId),
    getPostCount(userId),
    getFollowerCount(userId)
  ]);
  
  return { user, posts, followers };
}
```

---

## 11 Add Logging and Monitoring

Add logging to understand behavior after code is deployed.

**Logging Strategy:**

### Log Levels
```typescript
// ✅ Use appropriate log levels
logger.info('User registration started', { email, timestamp: Date.now() });

logger.warn('Cache miss for user profile', { userId, reason: 'TTL expired' });

logger.error('Database connection failed', { error, attempt: 3, connectionString: masked });

// For performance tracking
logger.debug('Query execution time', { query: 'SELECT users WHERE id = ?', duration: 42 });
```

### Add Performance Metrics
```typescript
// ✅ Log timing for potentially slow operations
const startTime = Date.now();
const results = await expensiveQuery();
const duration = Date.now() - startTime;

if (duration > 1000) {
  logger.warn('Slow query detected', { 
    query: 'user_profile', 
    duration,
    threshold: 1000 
  });
}
```

### Log Security Events
```typescript
// ✅ Log authentication and validation failures
logger.warn('Failed login attempt', {
  email,
  failureReason: 'invalid_password',
  attemptNumber: 3,
  ipAddress
});

logger.error('Validation failure for sensitive operation', {
  userId,
  operation: 'fund_transfer',
  amountCents,
  failureReason: 'insufficient_funds'
});
```

---

## 12 Apply DRY Principle

Each piece of knowledge should have a single, unambiguous representation.

**DRY Strategy:**

### Identify Repetitive Code
```typescript
// ❌ Duplicated validation logic
function createUser(email: string) {
  if (!email.includes('@')) throw new Error('Invalid email');
  // ...
}

function updateUser(id: number, email: string) {
  if (!email.includes('@')) throw new Error('Invalid email');
  // ...
}

// ✅ Extract to shared utility
function validateEmail(email: string): void {
  if (!email.includes('@')) throw new Error('Invalid email');
}

function createUser(email: string) {
  validateEmail(email);
  // ...
}

function updateUser(id: number, email: string) {
  validateEmail(email);
  // ...
}
```

### Use Inheritance and Composition
```typescript
// ✅ Base class for shared behavior
abstract class Entity {
  id: string;
  createdAt: Date;
  updatedAt: Date;
  
  abstract validate(): void;
  
  save(): Promise<void> {
    this.validate();
    this.updatedAt = new Date();
    // ... save logic
  }
}

class User extends Entity {
  email: string;
  
  validate(): void {
    if (!this.email.includes('@')) throw new Error('Invalid email');
  }
}
```

### Create Utility Classes
```typescript
// ✅ Shared utilities for common patterns
class ValidationUtils {
  static isValidEmail(email: string): boolean {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  }
  
  static isValidPhone(phone: string): boolean {
    return /^\+?[\d\s\-()]{10,}$/.test(phone);
  }
  
  static isValidUrl(url: string): boolean {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }
}
```

---

## 13 Code Organization and Structure

Organize code logically for easy navigation and maintenance.

**Best Practices:**

### Group Related Functionality
```
src/
├── users/
│   ├── user.model.ts           # User data model
│   ├── user.repository.ts      # Database access
│   ├── user.service.ts         # Business logic
│   ├── user.validator.ts       # Validation rules
│   ├── user.controller.ts      # HTTP handlers
│   └── user.test.ts            # Tests
├── shared/
│   ├── validators/
│   ├── utils/
│   └── types/
└── config/
```

### Co-locate Tests
```typescript
// ✅ Keep tests near implementation
// src/utils/validators.ts
export function validateEmail(email: string): boolean { ... }

// src/utils/validators.test.ts
describe('validators', () => {
  describe('validateEmail', () => {
    // tests here
  });
});
```

---

## 14 When Building, Follow This Checklist

Before considering code complete, verify:

- [ ] **Code Style:** Follows language conventions and project standards
- [ ] **Single Responsibility:** Each function/class does one thing well
- [ ] **Strong Typing:** Uses type system effectively
- [ ] **Input Validation:** Validates all inputs at function entry
- [ ] **Error Handling:** Appropriate try-catch and exception throwing
- [ ] **Comments:** Explain "why" for complex logic, not obvious "what"
- [ ] **DRY:** No unnecessary duplication of logic
- [ ] **Tests:** Comprehensive unit and integration tests with good coverage
- [ ] **Performance:** Uses efficient data structures and early filtering
- [ ] **Defensive:** Guards against unexpected inputs/states
- [ ] **Logging:** Includes appropriate logging and metrics
- [ ] **Documentation:** Clear docstrings and comments
- [ ] **No Hardcoding:** Configurable values, not magic numbers
- [ ] **Small Focused Changes:** Each commit is logically coherent

---

## Example: Building a User Registration Feature

Here's how these principles apply to a complete feature:

### Change 1: Core Validation Logic
```typescript
// src/users/user.validator.ts
export class UserValidator {
  static validateEmail(email: string): void {
    if (!email) throw new Error('Email is required');
    if (email.length > 254) throw new Error('Email is too long');
    if (!email.includes('@')) throw new Error('Email must contain @');
  }
  
  static validatePassword(password: string): void {
    if (!password) throw new Error('Password is required');
    if (password.length < 8) throw new Error('Password must be at least 8 characters');
    if (!/[A-Z]/.test(password)) throw new Error('Password must contain uppercase letter');
  }
}

// src/users/user.validator.test.ts
describe('UserValidator', () => {
  describe('validateEmail', () => {
    it('accepts valid emails', () => {
      expect(() => UserValidator.validateEmail('user@example.com')).not.toThrow();
    });
    
    it('rejects invalid emails', () => {
      expect(() => UserValidator.validateEmail('invalid')).toThrow('Email must contain @');
    });
  });
});
```

### Change 2: Repository Layer
```typescript
// src/users/user.repository.ts
export class UserRepository {
  async create(email: string, passwordHash: string): Promise<User> {
    console.assert(email, 'Email must be provided');
    console.assert(passwordHash, 'Password hash must be provided');
    
    const existing = await db.query(
      'SELECT id FROM users WHERE email = ?', 
      [email]
    );
    
    if (existing.length > 0) {
      throw new UserAlreadyExistsError(email);
    }
    
    const result = await db.query(
      'INSERT INTO users (email, passwordHash, createdAt) VALUES (?, ?, ?)',
      [email, passwordHash, new Date()]
    );
    
    return { id: result.insertId, email };
  }
}
```

### Change 3: Service Layer with Tests
```typescript
// src/users/user.service.ts
export class UserService {
  constructor(
    private userRepository: UserRepository,
    private passwordService: PasswordService,
    private logger: Logger
  ) {}
  
  async registerUser(email: string, password: string): Promise<User> {
    // Validate inputs early
    UserValidator.validateEmail(email);
    UserValidator.validatePassword(password);
    
    try {
      // Hash password with salt
      const passwordHash = await this.passwordService.hash(password);
      
      // Create user
      const user = await this.userRepository.create(email, passwordHash);
      
      this.logger.info('User registered successfully', { userId: user.id, email });
      
      return user;
    } catch (error) {
      if (error instanceof UserAlreadyExistsError) {
        this.logger.warn('Registration failed: user already exists', { email });
        throw new Error('Email already registered');
      }
      
      this.logger.error('Registration failed', { email, error });
      throw new Error('Failed to register user');
    }
  }
}

// src/users/user.service.test.ts
describe('UserService', () => {
  describe('registerUser', () => {
    it('creates user with valid credentials', async () => {
      const service = new UserService(mockRepository, mockPassword, mockLogger);
      
      const user = await service.registerUser('user@example.com', 'SecurePass123');
      
      expect(user.email).toBe('user@example.com');
      expect(mockRepository.create).toHaveBeenCalled();
    });
    
    it('rejects invalid email', async () => {
      const service = new UserService(mockRepository, mockPassword, mockLogger);
      
      await expect(
        service.registerUser('invalid', 'SecurePass123')
      ).rejects.toThrow('Email must contain @');
    });
  });
});
```

---

## Summary

When you ask Claude to build something, it will:

1. ✅ Follow strict coding style conventions
2. ✅ Break changes into small, focused commits
3. ✅ Write clean, defensive, modular code
4. ✅ Validate all inputs thoroughly
5. ✅ Assert invariants and assumptions
6. ✅ Comment to explain reasoning, not restate code
7. ✅ Handle errors appropriately for the context
8. ✅ Include comprehensive tests
9. ✅ Design clean, narrow APIs
10. ✅ Consider performance throughout
11. ✅ Add appropriate logging and monitoring
12. ✅ Follow DRY principles religiously

This ensures production-quality code that's easy to understand, maintain, extend, and debug.
