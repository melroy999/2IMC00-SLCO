/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.block_stmnt;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>block stmnt</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class block_stmntTest extends stmntTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(block_stmntTest.class);
	}

	/**
	 * Constructs a new block stmnt test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public block_stmntTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this block stmnt test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected block_stmnt getFixture() {
		return (block_stmnt)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createblock_stmnt());
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#tearDown()
	 * @generated
	 */
	@Override
	protected void tearDown() throws Exception {
		setFixture(null);
	}

} //block_stmntTest
